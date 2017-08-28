lib_LTLIBRARIES = libsentencepiece.la

AM_CXXFLAS = -I($srcdir)

libsentencepiece_la_SOURCES = \
	error.cc \
	flags.cc \
	sentencepiece_processor.cc \
	util.cc \
	normalizer.cc \
	stringpiece.h unicode_script_map.h util.h \
	common.h \
	flags.h  normalizer.h sentencepiece_processor.h  \
	model_factory.h model_factory.cc \
	model_interface.h model_interface.cc \
	unigram_model.h	unigram_model.cc  \
	word_model.h word_model.cc \
	char_model.h char_model.cc \
	bpe_model.h bpe_model.cc
include_HEADERS = sentencepiece_processor.h

noinst_LIBRARIES = libtrain.a
libtrain_a_SOURCES = builder.cc builder.h \
		    normalization_rule.h  \
	   	    unicode_script.h unicode_script.cc \
		    trainer_factory.h trainer_factory.cc \
		    trainer_interface.h trainer_interface.cc \
                    unigram_model_trainer.h unigram_model_trainer.cc \
	     	    word_model_trainer.h word_model_trainer.cc \
		    char_model_trainer.h char_model_trainer.cc \
	            bpe_model_trainer.h bpe_model_trainer.cc

nodist_libsentencepiece_la_SOURCES = \
	sentencepiece.pb.cc  sentencepiece.pb.h \
	sentencepiece_model.pb.cc sentencepiece_model.pb.h

BUILT_SOURCES = \
	sentencepiece.pb.cc \
	sentencepiece_model.pb.cc

EXTRA_DIST = sentencepiece.proto sentencepiece_model.proto

bin_PROGRAMS = spm_encode spm_decode spm_normalize spm_train spm_export_vocab
noinst_PROGRAMS = compile_charsmap

spm_encode_SOURCES = spm_encode_main.cc
spm_encode_LDADD = libsentencepiece.la

spm_decode_SOURCES = spm_decode_main.cc
spm_decode_LDADD = libsentencepiece.la

spm_normalize_SOURCES = spm_normalize_main.cc
spm_normalize_LDADD = libtrain.a libsentencepiece.la

spm_export_vocab_SOURCES = spm_export_vocab_main.cc
spm_export_vocab_LDADD = libsentencepiece.la

spm_train_SOURCES = spm_train_main.cc
spm_train_LDADD = libtrain.a libsentencepiece.la

compile_charsmap_SOURCES = compile_charsmap_main.cc
compile_charsmap_LDADD = libtrain.a libsentencepiece.la


check_PROGRAMS = spm_test
TESTS = spm_test
spm_test_SOURCES = testharness.h \
                   builder_test.cc \
	           flags_test.cc \
	           normalizer_test.cc \
	           sentencepiece_processor_test.cc \
	           unicode_script_test.cc \
	           model_interface_test.cc \
	           model_factory_test.cc \
	           trainer_interface_test.cc \
	           trainer_factory_test.cc \
	           word_model_test.cc \
	           word_model_trainer_test.cc \
		   bpe_model_test.cc \
		   bpe_model_trainer_test.cc \
	           char_model_test.cc \
	           char_model_trainer_test.cc \
	           unigram_model_test.cc\
	           unigram_model_trainer_test.cc \
                   util_test.cc \
	           test_main.cc \
                   testharness.cc

spm_test_LDADD = libtrain.a libsentencepiece.la

CLEANFILES = *.pb.cc *.pb.h *.pb.h *.gcda *.gcno *.info
clean-local:
	-rm -rf lcov_html

%.pb.cc %.pb.h: %.proto
	$(PROTOC) --cpp_out=$(srcdir) $(srcdir)/$<

coverage:
	make clean
	make -j10 CXXFLAGS+="-O0 -Wall -std=c++11 -coverage" LIBS+="-lgcov -lprotobuf" check
	lcov -c -d . -o coverage.info
	lcov --remove coverage.info "include*" "/c++" "_test*" "testharness*" "third_party*" ".pb.*" -o coverage.info
	mkdir -p lcov_html
	genhtml -o lcov_html coverage.info
