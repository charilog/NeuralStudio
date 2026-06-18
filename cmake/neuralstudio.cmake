# cmake/neuralstudio.cmake
# ─────────────────────────────────────────────────────────────────────────────
# Injected by CMake via:
#   -DCMAKE_PROJECT_INCLUDE:FILEPATH=$PWD/cmake/neuralstudio.cmake
#
# Runs as the LAST step of every project() call in this build tree,
# including CMake's internal try_compile test projects.
#
# ⚠  DO NOT set CMAKE_MSVC_RUNTIME_LIBRARY here.
#    Setting it in this file (with CACHE FORCE) causes it to be inherited
#    by CMake's internal TryCompile projects, which use Debug config
#    and reject generator-expression values evaluated at configure time.
#    The MSVC_RUNTIME_LIBRARY *target property* is set in CMakeLists.txt
#    after add_executable() where it only affects our own target.
# ─────────────────────────────────────────────────────────────────────────────

# ── Windows API surface ───────────────────────────────────────────────────────
if(WIN32)
    add_compile_definitions(
        WIN32_LEAN_AND_MEAN   # trim <windows.h> to the useful parts
        NOMINMAX              # prevent std::min/max macro clobbering
        UNICODE               # wide-char Win32 API
        _UNICODE
    )
endif()

# ── MSVC compiler flags ───────────────────────────────────────────────────────
if(MSVC)
    add_compile_options(
        /W3              # moderate warning level (Qt headers are noisy at /W4)
        /MP              # parallel compilation (use all available cores)
        /utf-8           # source files and string literals → UTF-8
        /permissive-     # ISO C++ strict conformance mode
        /Zc:__cplusplus  # report the real __cplusplus value  (Qt 6 requires it)
        /Zc:preprocessor # standards-conforming preprocessor  (Qt 6 requires it)
    )

    # Suppress warnings that fire inside Qt-generated moc_*.cpp code
    add_compile_options(
        /wd4127   # conditional expression is constant  (Q_ASSERT / Q_UNUSED)
        /wd4251   # class needs DLL interface            (Qt export macros)
        /wd4275   # non-DLL-interface base class         (Qt internals)
    )
endif()

# ── Output directories (multi-config: Release/ Debug/ …) ─────────────────────
# Visual Studio already puts outputs in build/<Config>/ by default.
# Explicitly mirror that for Ninja + other single-config generators.
foreach(_cfg Release Debug RelWithDebInfo MinSizeRel)
    string(TOUPPER ${_cfg} _CFG)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_CFG}
        "${CMAKE_BINARY_DIR}/${_cfg}" CACHE PATH "" FORCE)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${_CFG}
        "${CMAKE_BINARY_DIR}/${_cfg}" CACHE PATH "" FORCE)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${_CFG}
        "${CMAKE_BINARY_DIR}/${_cfg}" CACHE PATH "" FORCE)
endforeach()
