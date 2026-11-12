plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.google.sentencepiece"
    compileSdk = 34

    defaultConfig {
        minSdk = 21
        targetSdk = 34

        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17 -fexceptions -frtti"
                arguments = listOf(
                    "ANDROID_STL=c++_shared",
                    "ANDROID_TOOLCHAIN=clang"
                )
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    externalNativeBuild {
        cmake {
            path = "CMakeLists.txt"
        }
    }

    sourceSets {
        main {
            jniLibs.srcDirs("src/main/jniLibs")
            java.srcDirs("src/main/kotlin")
            res.srcDirs("src/main/res")
            manifest.srcFile("src/main/AndroidManifest.xml")
        }
    }
}

dependencies {
    implementation("org.jetbrains.kotlin:kotlin-stdlib:1.9.23")
}
