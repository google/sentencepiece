AUTOMAKE_OPTIONS = foreign
SUBDIRS = src

EXTRA_DIRS = m4 third_party data doc
EXTRA_DIST = README.md LICENSE
ACLOCAL_AMFLAGS = -I third_party/m4

dist-hook:
	for subdir in $(EXTRA_DIRS); do \
	  cp -rp $$subdir $(distdir); \
	  rm -f $(distdir)/$$subdir/*~; \
	  rm -f $(distdir)/$$subdir/*.{bak,orig}; \
	  rm -rf $(distdir)/$$subdir/CVS; \
	  rm -rf $(distdir)/$$subdir/.svn; \
	  rm -rf $(distdir)/.svn; \
	  rm -rf $(distdir)/*/.svn; \
	  rm -rf $(distdir)/*/*/.svn; \
	  rm -rf $(distdir)/$$subdir/*/CVS; \
	  rm -rf $(distdir)/$$subdir/*/.svn; \
  	  rm -rf $(distdir)/$$subdir/*/.pb.cc; \
	  find $(distdir) -name .svn  | xargs rm -fr; \
	done
