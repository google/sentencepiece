#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([sentencepiece], [0.1.0], [taku@google.com])
AM_INIT_AUTOMAKE()
AC_CONFIG_SRCDIR([src/normalizer.h])
AC_CONFIG_HEADERS([config.h])

# Checks for programs.
AC_LANG([C++])
AC_PROG_LIBTOOL
AC_PROG_CXX
AC_PROG_CC

CXXFLAGS="-std=c++11 -Wall -O3"

PKG_CHECK_MODULES(PROTOBUF, protobuf >= 2.4.0)
AC_SUBST(PROTOBUF_LIBS)
AC_SUBST(PROTOBUF_CFLAGS)
AC_SUBST(PROTOBUF_VERSION)
CXXFLAGS="$CXXFLAGS $PROTOBUF_CFLAGS"
LIBS="$LIBS $PROTOBUF_LIBS"

# Checks for header files.
AC_CHECK_HEADERS([unistd.h])

AC_CHECK_PROG([PROTOC], [protoc], [protoc])
AS_IF([test "x${PROTOC}" == "x"],
    [AC_MSG_ERROR([ProtoBuf compiler "protoc" not found. You can install them with "sudo apt-get install libprotobuf-c++ protobuf-compiler" ])])

AC_ARG_ENABLE(enable-nfkc-compile,
 [  --enable-nfkc-compile            compile NFKC normalizer mapping [default no]])
if test "$enable_nfkc_compile" = "yes"; then
AX_CHECK_ICU([40], ,AC_MSG_ERROR([Library requirements (ICU) not met.]))
CXXFLAGS="$CXXFLAGS -DENABLE_NFKC_COMPILE"
LIBS="$LIBS $ICU_LIBS"
fi

# Checks for typedefs, structures, and compiler characteristics.
AC_TYPE_SIZE_T

# Checks for library functions.
AC_FUNC_STRTOD
AC_CHECK_FUNCS([memchr memset])

AC_CONFIG_MACRO_DIR([third_party/m4])

AC_CONFIG_FILES([Makefile
                 src/Makefile])

AC_OUTPUT
