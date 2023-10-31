#ifndef MIXED_TEXT_CODE_HANDLER_H_
#define MIXED_TEXT_CODE_HANDLER_H_

#include <cassert>
#include "common.h"

namespace sentencepiece {

class MixedTextCodeIterator {
  const absl::string_view cache_value_;
  bool in_text_;
  const char* head_;
  const char* tail_;
  const int32 verbatim_control_char_;
  const int32 code_block_end_;
  const int32 code_meta_block_begin_;
  const int32 code_meta_block_end_;

  bool HasCodeHeader() const;

  bool ReadCodeHeader(absl::string_view* line);

  bool ReadTextBlock(absl::string_view* line);

  bool ReadCodeBlock(absl::string_view* line);

  bool TryReadNext(absl::string_view* line);

public:
  MixedTextCodeIterator(absl::string_view cache_value,
    int32 verbatim_control_char,
    int32 code_block_end,
    int32 code_meta_block_begin,
    int32 code_meta_block_end
  );

  bool Next(absl::string_view* line);

  bool HasNext() const;
};

}  // namespace sentencepiece
#endif  // TRAINER_INTERFACE_H_