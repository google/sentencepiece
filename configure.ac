#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([sentencepiece], [0.1.0], [taku@google.com])
AM_INIT_AUTOMAKE()
AC_CONFIG_SRCDIR([src/normalizer.h])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([third_party/m4])

# Checks for programs.
AC_LANG([C++])
AC_PROG_LIBTOOL
AC_PROG_CXX
AC_PROG_CC

CXXFLAGS="-std=c++11 -Wall -O3 -fPIC"

PKG_CHECK_MODULES(PROTOBUF, protobuf >= 2.4.0)
AC_SUBST(PROTOBUF_LIBS)
AC_SUBST(PROTOBUF_CFLAGS)
AC_SUBST(PROTOBUF_VERSION)
CXXFLAGS="$CXXFLAGS $PROTOBUF_CFLAGS"
LIBS="$LIBS $PROTOBUF_LIBS"

m4_include([third_party/m4/ax_check_icu.m4])

if test ! -n "$PROTOC"; then
  AC_CHECK_PROG([PROTOC], [protoc], [protoc])
fi
AS_IF([test "x${PROTOC}" == "x"],
    [AC_MSG_ERROR([ProtoBuf compiler "protoc" not found. You can install them with "sudo apt-get install libprotobuf-c++ protobuf-compiler" ])])

# --enable-nfkc-compile flag.
AC_MSG_CHECKING([nfkc-compile option])
AC_ARG_ENABLE([nfkc-compile],
              [AS_HELP_STRING([--enable-nfkc-compile], [compile NFKC normalizer mapping])],
              [],
              [enable_nfkc_compile=no])
AC_MSG_RESULT([$enable_nfkc_compile])

if test "${enable_nfkc_compile}" = "yes"; then
  AX_CHECK_ICU([40], [], AC_MSG_ERROR([Library requirements (ICU) not met.]))
  CXXFLAGS="$CXXFLAGS -DENABLE_NFKC_COMPILE"
  LIBS="$LIBS $ICU_LIBS"
fi

# --enable-gcov flag.
AC_MSG_CHECKING([gcov option])
AC_ARG_ENABLE([gcov],
              [AS_HELP_STRING([--enable-gcov], [generate enable-gcov files])],
              [],
              [enable_gcov=no])
AC_MSG_RESULT([$enable_gcov])

if test "${enable_gcov}" = "yes"; then
  CXXFLAGS="-std=c++11 -Wall -fPIC -fprofile-arcs -O0 -coverage"
  LIBS="$LIBS -lgcov"
fi

# pkgconfigdir
AC_ARG_WITH(pkgconfigdir,
	AC_HELP_STRING([--with-pkgconfigdir],
	[Use the specified pkgconfig dir (default is libdir/pkgconfig)]),
	[pkgconfigdir=${withval}],
	[pkgconfigdir='${libdir}/pkgconfig'])
AC_MSG_NOTICE([pkgconfig directory is ${pkgconfigdir}])
pkgconfigcflags=$CFLAGS
pkgconfiglibs=$LIBS
AC_SUBST([pkgconfigdir])
AC_SUBST([pkgconfigcflags])
AC_SUBST([pkgconfiglibs])

# Checks for header files.
AC_CHECK_HEADERS([unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_TYPE_SIZE_T

# Checks for library functions.
AC_FUNC_STRTOD
AC_CHECK_FUNCS([memchr memset])

AC_CONFIG_FILES([Makefile
                 src/Makefile
		 sentencepiece.pc])

AC_OUTPUT
