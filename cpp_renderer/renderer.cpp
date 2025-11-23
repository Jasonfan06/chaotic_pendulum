#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr uint32_t kMagic = 0x43485044; // "CHPD"
constexpr uint32_t kVersion = 1;

struct FileHeader {
    uint32_t magic{};
    uint32_t version{};
    uint32_t frame_count{};
    uint32_t node_count{};
    uint32_t instance_count{};
};

struct Color {
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

struct Options {
    fs::path input_path;
    fs::path output_path{"cpp_renderer_output.mp4"};
    int width{1920};
    int height{1080};
    double fps{60.0};
    double playback_speed{1.0};
    double axis_limit{0.0};
    double line_alpha{0.45};
    double point_alpha{0.85};
    double background_brightness{0.05};
};

template <typename T>
T clamp(T value, T min_value, T max_value) {
    return std::max(min_value, std::min(value, max_value));
}

struct Framebuffer {
    int width;
    int height;
    std::vector<uint8_t> pixels;

    explicit Framebuffer(int w, int h)
        : width(w), height(h), pixels(static_cast<size_t>(w) * static_cast<size_t>(h) * 3u, 0) {}

    void clear(float brightness) {
        const uint8_t value = static_cast<uint8_t>(clamp(brightness, 0.0f, 1.0f) * 255.0f);
        std::fill(pixels.begin(), pixels.end(), value);
    }

    void blend_pixel(int x, int y, const Color& color, float alpha) {
        if (x < 0 || x >= width || y < 0 || y >= height) {
            return;
        }
        alpha = clamp(alpha, 0.0f, 1.0f);
        const size_t index = static_cast<size_t>(y) * static_cast<size_t>(width) * 3ull + static_cast<size_t>(x) * 3ull;
        const std::array<uint8_t, 3> src{color.r, color.g, color.b};
        for (size_t i = 0; i < 3; ++i) {
            const float dst = static_cast<float>(pixels[index + i]);
            const float blended = dst * (1.0f - alpha) + static_cast<float>(src[i]) * alpha;
            pixels[index + i] = static_cast<uint8_t>(clamp(blended, 0.0f, 255.0f));
        }
    }

    void draw_line(int x0, int y0, int x1, int y1, const Color& color, float alpha) {
        x0 = clamp(x0, 0, width - 1);
        x1 = clamp(x1, 0, width - 1);
        y0 = clamp(y0, 0, height - 1);
        y1 = clamp(y1, 0, height - 1);

        int dx = std::abs(x1 - x0);
        int sx = x0 < x1 ? 1 : -1;
        int dy = -std::abs(y1 - y0);
        int sy = y0 < y1 ? 1 : -1;
        int err = dx + dy;

        while (true) {
            blend_pixel(x0, y0, color, alpha);
            if (x0 == x1 && y0 == y1) {
                break;
            }
            const int e2 = err * 2;
            if (e2 >= dy) {
                err += dy;
                x0 += sx;
            }
            if (e2 <= dx) {
                err += dx;
                y0 += sy;
            }
        }
    }

    void draw_disk(int cx, int cy, int radius, const Color& color, float alpha) {
        if (radius <= 0) {
            blend_pixel(cx, cy, color, alpha);
            return;
        }
        const int r2 = radius * radius;
        for (int y = -radius; y <= radius; ++y) {
            for (int x = -radius; x <= radius; ++x) {
                if (x * x + y * y <= r2) {
                    blend_pixel(cx + x, cy + y, color, alpha);
                }
            }
        }
    }
};

struct Projector {
    double min_x;
    double max_x;
    double min_y;
    double max_y;
    int width;
    int height;

    std::pair<int, int> project(double x, double y) const {
        const double nx = (x - min_x) / (max_x - min_x);
        const double ny = (y - min_y) / (max_y - min_y);
        const int px = static_cast<int>(clamp(nx, 0.0, 1.0) * static_cast<double>(width - 1));
        const int py = static_cast<int>((1.0 - clamp(ny, 0.0, 1.0)) * static_cast<double>(height - 1));
        return {px, py};
    }
};

struct Dataset {
    std::vector<float> x;
    std::vector<float> y;
    uint32_t frames{};
    uint32_t nodes{};
    uint32_t instances{};
};

Options parse_args(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg{argv[i]};
        auto require_value = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + flag);
            }
            return std::string(argv[++i]);
        };

        if (arg == "--input") {
            opts.input_path = require_value(arg);
        } else if (arg == "--output") {
            opts.output_path = require_value(arg);
        } else if (arg == "--width") {
            opts.width = std::stoi(require_value(arg));
        } else if (arg == "--height") {
            opts.height = std::stoi(require_value(arg));
        } else if (arg == "--fps") {
            opts.fps = std::stod(require_value(arg));
        } else if (arg == "--speed") {
            opts.playback_speed = std::stod(require_value(arg));
        } else if (arg == "--axis-limit") {
            opts.axis_limit = std::stod(require_value(arg));
        } else if (arg == "--line-alpha") {
            opts.line_alpha = std::stod(require_value(arg));
        } else if (arg == "--point-alpha") {
            opts.point_alpha = std::stod(require_value(arg));
        } else if (arg == "--background") {
            opts.background_brightness = std::stod(require_value(arg));
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: pendulum_renderer --input payload.bin --output video.mp4 [options]\n"
                         "Options:\n"
                         "  --width <int>         Video width (default 1920)\n"
                         "  --height <int>        Video height (default 1080)\n"
                         "  --fps <float>         Frames per second (default 60)\n"
                         "  --speed <float>       Playback speed multiplier (default 1.0)\n"
                         "  --axis-limit <float>  Scene half-width (default based on N)\n"
                         "  --line-alpha <float>  Line alpha (0-1)\n"
                         "  --point-alpha <float> Point alpha (0-1)\n"
                         "  --background <float>  Background brightness (0-1)\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (opts.input_path.empty()) {
        throw std::runtime_error("--input is required");
    }
    if (opts.output_path.empty()) {
        throw std::runtime_error("--output is required");
    }
    if (opts.width <= 0 || opts.height <= 0) {
        throw std::runtime_error("Width and height must be positive");
    }
    if (opts.fps <= 0.0) {
        throw std::runtime_error("FPS must be positive");
    }
    if (opts.playback_speed <= 0.0) {
        throw std::runtime_error("Speed must be positive");
    }
    opts.line_alpha = clamp(opts.line_alpha, 0.0, 1.0);
    opts.point_alpha = clamp(opts.point_alpha, 0.0, 1.0);
    opts.background_brightness = clamp(opts.background_brightness, 0.0, 1.0);
    return opts;
}

Dataset load_dataset(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open input payload: " + path.string());
    }

    FileHeader header{};
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!file) {
        throw std::runtime_error("Failed to read header");
    }
    if (header.magic != kMagic) {
        throw std::runtime_error("Invalid payload: magic mismatch");
    }
    if (header.version != kVersion) {
        throw std::runtime_error("Unsupported payload version");
    }

    const uint64_t total = static_cast<uint64_t>(header.frame_count) *
                           static_cast<uint64_t>(header.node_count) *
                           static_cast<uint64_t>(header.instance_count);

    Dataset data;
    data.frames = header.frame_count;
    data.nodes = header.node_count;
    data.instances = header.instance_count;
    data.x.resize(total);
    data.y.resize(total);

    file.read(reinterpret_cast<char*>(data.x.data()), static_cast<std::streamsize>(total * sizeof(float)));
    if (!file) {
        throw std::runtime_error("Failed to read X positions");
    }
    file.read(reinterpret_cast<char*>(data.y.data()), static_cast<std::streamsize>(total * sizeof(float)));
    if (!file) {
        throw std::runtime_error("Failed to read Y positions");
    }

    return data;
}

std::string shell_escape(const fs::path& path) {
    std::ostringstream oss;
    oss << '"';
    for (const char c : path.string()) {
        if (c == '"') {
            oss << "\\\"";
        } else {
            oss << c;
        }
    }
    oss << '"';
    return oss.str();
}

FILE* open_ffmpeg(const Options& opts) {
    const double render_fps = opts.fps * opts.playback_speed;
    std::ostringstream cmd;
    cmd << "ffmpeg -loglevel warning -y -f rawvideo -pix_fmt rgb24 -s "
        << opts.width << "x" << opts.height
        << " -r " << render_fps
        << " -i - -an -c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p "
        << shell_escape(opts.output_path);

    FILE* pipe = popen(cmd.str().c_str(), "w");
    if (!pipe) {
        throw std::runtime_error("Failed to launch ffmpeg. Is it installed?");
    }
    return pipe;
}

void close_ffmpeg(FILE* pipe) {
    if (pipe) {
        const int code = pclose(pipe);
        if (code != 0) {
            throw std::runtime_error("ffmpeg exited with code " + std::to_string(code));
        }
    }
}

inline uint64_t index(uint32_t frame, uint32_t node, uint32_t instance,
                      uint32_t nodes, uint32_t instances) {
    return static_cast<uint64_t>(frame) * nodes * instances +
           static_cast<uint64_t>(node) * instances +
           static_cast<uint64_t>(instance);
}

void render_frames(const Dataset& data, const Options& opts) {
    Framebuffer framebuffer(opts.width, opts.height);

    const double axis_limit =
        opts.axis_limit > 0.0 ? opts.axis_limit : std::max(1.0, static_cast<double>(data.nodes - 1)) * 1.2;
    const Projector projector{
        -axis_limit,
        axis_limit,
        -axis_limit,
        axis_limit * 0.25,
        opts.width,
        opts.height,
    };

    const Color line_color{255, 255, 255};
    const Color point_color{255, 230, 60};
    const Color origin_color{255, 60, 60};

    FILE* ffmpeg = nullptr;
    try {
        ffmpeg = open_ffmpeg(opts);
        const float bg = static_cast<float>(opts.background_brightness);
        const float line_alpha = static_cast<float>(opts.line_alpha);
        const float point_alpha = static_cast<float>(opts.point_alpha);
        const int point_radius = std::max(2, opts.height / 180);
        const int origin_radius = std::max(3, opts.height / 140);

        for (uint32_t frame = 0; frame < data.frames; ++frame) {
            framebuffer.clear(bg);

            for (uint32_t instance = 0; instance < data.instances; ++instance) {
                int prev_x = 0;
                int prev_y = 0;
                bool has_prev = false;

                for (uint32_t node = 0; node < data.nodes; ++node) {
                    const uint64_t idx = index(frame, node, instance, data.nodes, data.instances);
                    const double x = static_cast<double>(data.x[idx]);
                    const double y = static_cast<double>(data.y[idx]);
                    const auto [px, py] = projector.project(x, y);

                    if (has_prev) {
                        framebuffer.draw_line(prev_x, prev_y, px, py, line_color, line_alpha);
                    }

                    framebuffer.draw_disk(px, py, point_radius, point_color, point_alpha);
                    prev_x = px;
                    prev_y = py;
                    has_prev = true;
                }
            }

            const auto [ox, oy] = projector.project(0.0, 0.0);
            framebuffer.draw_disk(ox, oy, origin_radius, origin_color, 0.9f);

            const size_t written = fwrite(framebuffer.pixels.data(), 1, framebuffer.pixels.size(), ffmpeg);
            if (written != framebuffer.pixels.size()) {
                throw std::runtime_error("Failed to write frame to ffmpeg");
            }

            if ((frame + 1) % 60 == 0 || frame + 1 == data.frames) {
                const double progress = static_cast<double>(frame + 1) / static_cast<double>(data.frames) * 100.0;
                std::cout << "\rRendering: " << static_cast<int>(progress) << "%";
                std::cout.flush();
            }
        }

        std::cout << "\rRendering: 100%\n";
        close_ffmpeg(ffmpeg);
    } catch (...) {
        if (ffmpeg) {
            pclose(ffmpeg);
        }
        throw;
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options opts = parse_args(argc, argv);
        const Dataset data = load_dataset(opts.input_path);
        if (data.frames == 0 || data.nodes == 0 || data.instances == 0) {
            throw std::runtime_error("Payload is empty");
        }
        render_frames(data, opts);
        std::cout << "Video written to " << opts.output_path << std::endl;
        return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
}

