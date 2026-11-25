// logline.h: every line hyprp prints starts with the UTC time and a level, as in Mass Prime Siever's log:
//   2026-11-25 06:20:01 UTC+0 [INFO] Gerbicz check OK at iteration 1000000
// INFO and WARN go to stdout, FATAL (the run stops) to stderr; each line is flushed at once.
#pragma once
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>

inline void logLine(FILE* f, const char* level, const char* fmt, va_list ap) {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tm);
    std::fprintf(f, "%s UTC+0 [%s] ", ts, level);
    std::vfprintf(f, fmt, ap);
    std::fputc('\n', f);
    std::fflush(f);
}
inline void logInfo(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    logLine(stdout, "INFO", fmt, ap);
    va_end(ap);
}
inline void logWarn(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    logLine(stdout, "WARN", fmt, ap);
    va_end(ap);
}
inline void logFatal(const char* fmt, ...) {
    std::fflush(stdout);
    va_list ap;
    va_start(ap, fmt);
    logLine(stderr, "FATAL", fmt, ap);
    va_end(ap);
}
// the message part of a line hyprp printed (the line itself if it has no "YYYY-MM-DD HH:MM:SS UTC+0 [LEVEL] " prefix)
inline const char* logMessage(const char* line) {
    if (std::strlen(line) > 27 && std::strncmp(line + 19, " UTC+0 [", 8) == 0) {
        const char* e = std::strchr(line + 27, ']');
        if (e && e[1] == ' ') return e + 2;
    }
    return line;
}
