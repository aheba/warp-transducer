# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/conda/bin/cmake

# The command to remove a file.
RM = /opt/conda/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /network/tmp1/abdelwah/singularity_ngc/warp-transducer-modified

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /network/tmp1/abdelwah/singularity_ngc/warp-transducer-modified/build

# Include any dependencies generated for this target.
include CMakeFiles/test_time.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test_time.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_time.dir/flags.make

CMakeFiles/test_time.dir/tests/test_time.cpp.o: CMakeFiles/test_time.dir/flags.make
CMakeFiles/test_time.dir/tests/test_time.cpp.o: ../tests/test_time.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/network/tmp1/abdelwah/singularity_ngc/warp-transducer-modified/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_time.dir/tests/test_time.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_time.dir/tests/test_time.cpp.o -c /network/tmp1/abdelwah/singularity_ngc/warp-transducer-modified/tests/test_time.cpp

CMakeFiles/test_time.dir/tests/test_time.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_time.dir/tests/test_time.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /network/tmp1/abdelwah/singularity_ngc/warp-transducer-modified/tests/test_time.cpp > CMakeFiles/test_time.dir/tests/test_time.cpp.i

CMakeFiles/test_time.dir/tests/test_time.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_time.dir/tests/test_time.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /network/tmp1/abdelwah/singularity_ngc/warp-transducer-modified/tests/test_time.cpp -o CMakeFiles/test_time.dir/tests/test_time.cpp.s

CMakeFiles/test_time.dir/tests/random.cpp.o: CMakeFiles/test_time.dir/flags.make
CMakeFiles/test_time.dir/tests/random.cpp.o: ../tests/random.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/network/tmp1/abdelwah/singularity_ngc/warp-transducer-modified/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/test_time.dir/tests/random.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_time.dir/tests/random.cpp.o -c /network/tmp1/abdelwah/singularity_ngc/warp-transducer-modified/tests/random.cpp

CMakeFiles/test_time.dir/tests/random.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_time.dir/tests/random.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /network/tmp1/abdelwah/singularity_ngc/warp-transducer-modified/tests/random.cpp > CMakeFiles/test_time.dir/tests/random.cpp.i

CMakeFiles/test_time.dir/tests/random.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_time.dir/tests/random.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /network/tmp1/abdelwah/singularity_ngc/warp-transducer-modified/tests/random.cpp -o CMakeFiles/test_time.dir/tests/random.cpp.s

# Object files for target test_time
test_time_OBJECTS = \
"CMakeFiles/test_time.dir/tests/test_time.cpp.o" \
"CMakeFiles/test_time.dir/tests/random.cpp.o"

# External object files for target test_time
test_time_EXTERNAL_OBJECTS =

test_time: CMakeFiles/test_time.dir/tests/test_time.cpp.o
test_time: CMakeFiles/test_time.dir/tests/random.cpp.o
test_time: CMakeFiles/test_time.dir/build.make
test_time: libwarprnnt.so
test_time: /usr/local/cuda/lib64/libcudart_static.a
test_time: /usr/lib/x86_64-linux-gnu/librt.so
test_time: CMakeFiles/test_time.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/network/tmp1/abdelwah/singularity_ngc/warp-transducer-modified/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable test_time"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_time.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_time.dir/build: test_time

.PHONY : CMakeFiles/test_time.dir/build

CMakeFiles/test_time.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_time.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_time.dir/clean

CMakeFiles/test_time.dir/depend:
	cd /network/tmp1/abdelwah/singularity_ngc/warp-transducer-modified/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /network/tmp1/abdelwah/singularity_ngc/warp-transducer-modified /network/tmp1/abdelwah/singularity_ngc/warp-transducer-modified /network/tmp1/abdelwah/singularity_ngc/warp-transducer-modified/build /network/tmp1/abdelwah/singularity_ngc/warp-transducer-modified/build /network/tmp1/abdelwah/singularity_ngc/warp-transducer-modified/build/CMakeFiles/test_time.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_time.dir/depend

