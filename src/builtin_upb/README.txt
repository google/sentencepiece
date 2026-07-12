How to generate upb stub files:

1. Install protobuf compiler (protoc) and upb compiler (upbc).
   See: https://github.com/protocolbuffers/upb

2. Run the following commands to generate stub files from .proto files:

   # From this directory:
   protoc -I../ --upb_out=. ../sentencepiece.proto
   protoc -I../ --upb_out=. ../sentencepiece_model.proto

   # If you need upbdefs (not used by default, but kept for compatibility):
   # protoc -I../ --upbdefs_out=. ../sentencepiece.proto
   # protoc -I../ --upbdefs_out=. ../sentencepiece_model.proto

   (Note: depending on the upb version, you may need to specify the plugin path:
    --plugin=protoc-gen-upb=/path/to/upbc)

These generated files (*.upb.h, *.upb.c) are checked into VCS
to allow building without upb compiler dependency.
