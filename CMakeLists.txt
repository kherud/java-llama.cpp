cmake_minimum_required(VERSION 3.12)

project(jllama CXX)

set(LLAMA_CPP_TAG b1645)

# todo: Is there a better way to build the library than copy & pasting the build argument cmake definition of llama.cpp?
include(build-args.cmake)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS ON)

find_package(Java REQUIRED)
find_program(JAVA_EXECUTABLE NAMES java)

# find "jni.h" include directory
find_path(JNI_INCLUDE_DIR NAMES jni.h HINTS ENV JAVA_HOME PATH_SUFFIXES include)
if(NOT JNI_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find jni.h")
endif()

# find "jni_md.h" include directory if not set
file(GLOB_RECURSE JNI_MD_PATHS RELATIVE "${JNI_INCLUDE_DIR}" "${JNI_INCLUDE_DIR}/**/jni_md.h")
if(NOT JNI_MD_PATHS)
    message(FATAL_ERROR "Could not find jni_md.h")
endif()
foreach(PATH IN LISTS JNI_MD_PATHS)
    get_filename_component(DIR ${PATH} DIRECTORY)
    list(APPEND JNI_MD_INCLUDE_DIRS "${JNI_INCLUDE_DIR}/${DIR}")
endforeach()

# find which OS we build for if not set (make sure to run mvn compile first)
if(NOT DEFINED OS_NAME)
	execute_process(
      COMMAND ${JAVA_EXECUTABLE} -cp ${CMAKE_SOURCE_DIR}/target/classes de.kherud.llama.OSInfo --os
      OUTPUT_VARIABLE OS_NAME
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()
if(NOT OS_NAME)
    message(FATAL_ERROR "Could not determine OS name")
endif()

# find which architecture we build for if not set  (make sure to run mvn compile first)
if(NOT DEFINED OS_ARCH)
    execute_process(
      COMMAND ${JAVA_EXECUTABLE} -cp ${CMAKE_SOURCE_DIR}/target/classes de.kherud.llama.OSInfo --arch
      OUTPUT_VARIABLE OS_ARCH
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()
if(NOT OS_ARCH)
    message(FATAL_ERROR "Could not determine CPU architecture")
endif()

set(JLLAMA_DIR ${CMAKE_SOURCE_DIR}/src/main/resources/de/kherud/llama/${OS_NAME}/${OS_ARCH})
message(STATUS "Installing files to ${JLLAMA_DIR}")

# checkout llama.cpp
include(FetchContent)
FetchContent_Declare(
	llama.cpp
	GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
	GIT_TAG        ${LLAMA_CPP_TAG}
)
FetchContent_MakeAvailable(llama.cpp)

add_library(jllama SHARED src/main/cpp/jllama.cpp)

target_include_directories(jllama PRIVATE
    ${JNI_INCLUDE_DIR}
    ${JNI_MD_INCLUDE_DIRS}
    src/main/cpp
)

target_link_libraries(jllama PRIVATE common llama ${LLAMA_EXTRA_LIBS})

target_compile_features(jllama PRIVATE cxx_std_11)

if(OS_NAME STREQUAL "Windows")
	set_target_properties(jllama llama PROPERTIES
	  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${JLLAMA_DIR}
	)
else()
	set_target_properties(jllama llama PROPERTIES
	  LIBRARY_OUTPUT_DIRECTORY ${JLLAMA_DIR}
	)
endif()

if (LLAMA_METAL)
    # copy ggml-metal.metal to bin directory
    configure_file(${llama.cpp_SOURCE_DIR}/ggml-metal.metal ${JLLAMA_DIR}/ggml-metal.metal COPYONLY)
endif()
