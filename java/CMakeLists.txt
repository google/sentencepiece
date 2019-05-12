find_package(JNI)

set(SPM_JAVA_SRCS
        com_github_google_sentencepiece_SentencePieceJNI.h
        com_github_google_sentencepiece_SentencePieceJNI.cc)

if (JNI_FOUND)
    add_library(sentencepiece_java SHARED ${SPM_JAVA_SRCS})
    target_include_directories(sentencepiece_java PUBLIC ${JNI_INCLUDE_DIRS})
    target_link_libraries(sentencepiece_java sentencepiece-static)

    install(TARGETS sentencepiece_java
            LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/src/main/resources)
endif ()
