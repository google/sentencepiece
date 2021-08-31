// Copyright 2016 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.!

#ifndef NORMALIZER_CASE_ENCODER_H_
#define NORMALIZER_CASE_ENCODER_H_

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <deque>

#include "common.h"
#include "third_party/absl/strings/string_view.h"


namespace sentencepiece {
namespace normalizer {

std::vector<std::pair<const char*, const char*>> search(const std::string& input);

constexpr char cUppercase    = 'U';
constexpr char cAllUppercase = 'A';
constexpr char cTitlecase    = 'T';
constexpr char cLowercase    = 'L';
constexpr char cPunctuation  = 'P';
constexpr char cSpace        = ' ';

class CaseEncoder {
protected:
  typedef std::function<std::pair<absl::string_view, int>(absl::string_view)> Normalizer;
  Normalizer normalizer_;

public:
  virtual ~CaseEncoder() {}
  
  virtual std::pair<absl::string_view, int> normalizePrefix(absl::string_view input) {
    return normalizer_(input);
  }

  virtual void setNormalizer(Normalizer normalizer) {
    normalizer_ = normalizer;
  }

  virtual void postProcess(std::string* normalized, std::vector<size_t>* norm_to_orig) {}

  static std::unique_ptr<CaseEncoder> Create(bool, bool, bool);
};

class UpperCaseEncoder : public CaseEncoder {
private:
  std::string buffer_;
  std::string signature_;
  int offset_ = 0;

  // This is a queue consisting of all buffered "chars" and
  // the corresponding number of consumed bytes
  std::vector<std::pair<std::string, int> > buffer_queue_;
  int dump_buffer_from_ = -1;
  
  int state_{0};
  size_t spans_{0};
  bool seenThreeSpans_{false};
  bool removeExtraWhiteSpace_{false};

public:
  UpperCaseEncoder(bool removeExtraWhiteSpace)
  : removeExtraWhiteSpace_(removeExtraWhiteSpace) {}

  std::pair<absl::string_view, int> normalizePrefix(absl::string_view orig_input) {
    // dump_buffer_from controls the return process for characters and consumed bytes
    // When this is -1, we are in "collection" mode and just keep adding to the buffer.
    // When collection is complete at some logical point, we set this to 0 to force an
    // element-wise buffer dump until it is exhausted. The following "if" block is
    // responsible for this.
    if((dump_buffer_from_ >= 0) && (dump_buffer_from_ < buffer_queue_.size())) {
      return buffer_queue_[dump_buffer_from_++];
    }

    // Since the buffer is exhausted, we reset the dump_buffer_from_ flag to -1
    // indicating that we are now again in character "collection" mode.
    if(dump_buffer_from_ > -1) {
      dump_buffer_from_ = -1;
      buffer_queue_.clear();

      // This just results in the normalizePrefix function getting called again with
      // the same arguments by the caller.
      return {nullptr, 0};
    }

    const auto input = orig_input.substr(offset_);
    auto p = CaseEncoder::normalizePrefix(input);
    auto sp = p.first;
    int consumed = p.second;

    bool last = input.size() == (size_t)consumed;
    decltype(p) ret;

    // Calling this results in returning the empty token with a zero consumed
    // count that essentially results in normalizePrefix being called again with
    // exactly same arguments. However, we record the number of bytes consumed in
    // an offset so our view of the string does move with the correct amount.
    // We need to do this because returning an empty character, and a non-empty
    // consumed count will result in the caller messing up the norm_to_orig
    // mapping that it tracks.
    auto null = [this](int consumed) -> std::pair<absl::string_view, int> {
      offset_ += consumed;
      return {{nullptr, 0}, 0};
    };

    // This function is responsible for collecting a character and corresponding
    // number of consumed bytes in the buffer. We do _not_ want to return the characters
    // processed at this time, and instead just record the current state, and let the caller
    // call us again with the exact same input after the internal state is transitioned.
    // Once we are in the terminal state, we can dump the buffer collected by this point.
    auto buffer = [this, p](absl::string_view sp, int override_consumed = -1) {
      auto cur_buf_last = buffer_.size();
      buffer_.append(sp.data(), sp.size());
      auto tmp_str = std::string(buffer_).substr(cur_buf_last, sp.size());
      // ASSERT: concat([buffer_queue[0]]) == buffer_
      buffer_queue_.push_back({tmp_str, override_consumed == -1 ? p.second : override_consumed});
    };

    auto isUpper  = [=](absl::string_view sp) { return sp[0] == cUppercase;   };
    auto isPunct  = [=](absl::string_view sp) { return sp[0] == cPunctuation; };
    auto isSpace  = [=](absl::string_view sp) { return sp[0] == ' '; };

    if(state_ == 0) {
      buffer_.clear();
      buffer_queue_.clear();
      offset_ = 0;
    }

    if(isUpper(sp)) {
      if(state_ == 0) {
        buffer(sp);
        buffer_[0] = cTitlecase;
        buffer_queue_.front().first[0] = cTitlecase;
        
        state_ = 1;
        ret = null(consumed);
        
        signature_.append("U");
        signature_.append(sp.size() - 1, 'u');

      } else if(state_ == 1 || state_ == 2) {
        if(state_ == 1)
          spans_++;
        
        sp.remove_prefix(1);
        buffer(sp);
        buffer_queue_.front().first[0] = cUppercase;
        buffer_[0] = cUppercase;
        state_ = 2;
        ret = null(consumed);

        signature_.append(sp.size(), 'u');
      }  

      if(last) {
        dump_buffer_from_ = 0;
        return null(0);
      }
    } else {
      if(isPunct(sp)) {
        if(state_ == 1)
          spans_++;

        sp.remove_prefix(1);
        signature_.append(sp.size(), 'p');
      } else if(state_ == 2 && !isSpace(sp)) {
        spans_ = 0;
        buffer(std::string(1, cLowercase), 0);
        signature_.append("L");
        signature_.append(sp.size(), 'l');
      } else if(isSpace(sp)) {
        if(state_ == 1)
          spans_++;
        if(!removeExtraWhiteSpace_ || signature_.empty() || signature_.back() != 's')
          signature_.append("sss");
      } else {
        spans_ = 0;
        signature_.append(sp.size(), 'l');
      }

      if(!buffer_.empty()) {
        buffer(sp);
        offset_ = 0;
        dump_buffer_from_ = 0;
        state_ = 0;
        return null(0);
      } else {
        p.first = sp;
      }
      
      state_ = 0;
      ret = p;
    }

    if(spans_ >= 3)
      seenThreeSpans_ = true;

    return ret;
  }

  virtual void postProcess(std::string* normalized, std::vector<size_t>* norm_to_orig) { 
    if(!seenThreeSpans_)
      return;

    std::string normalized_temp;
    normalized_temp.reserve(normalized->size());
    
    std::vector<size_t> norm_to_orig_temp;
    norm_to_orig_temp.reserve(norm_to_orig->size());

    const char* sig_it = signature_.data();

    auto nrm_it = normalized->cbegin();
    auto n2o_it = norm_to_orig->cbegin();

    for(const auto& span : search(signature_)) {
      size_t len = std::distance(sig_it, span.first);
      
      normalized_temp.insert(normalized_temp.end(), nrm_it, nrm_it + len);
      norm_to_orig_temp.insert(norm_to_orig_temp.end(), n2o_it, n2o_it + len);

      sig_it += len; 
      nrm_it += len;
      n2o_it += len;
      normalized_temp.push_back(cAllUppercase);
      norm_to_orig_temp.push_back(*n2o_it);
            
      while(sig_it != span.second) {
        if(*sig_it == cUppercase) {
          sig_it++; 
          nrm_it++;
          n2o_it++;
        }
        sig_it++;
        normalized_temp.push_back(*nrm_it++);
        norm_to_orig_temp.push_back(*n2o_it++);
      }
      if(sig_it != signature_.data() + signature_.length()) { 
        if(*sig_it != cUppercase) {
          normalized_temp.push_back(cLowercase);
          norm_to_orig_temp.push_back(*n2o_it);
        }
      }
    }

    if(nrm_it != normalized->cend())
      normalized_temp.insert(normalized_temp.end(), nrm_it, normalized->cend());
    if(n2o_it != norm_to_orig->cend())
      norm_to_orig_temp.insert(norm_to_orig_temp.end(), n2o_it, norm_to_orig->cend());

    normalized->swap(normalized_temp);
    norm_to_orig->swap(norm_to_orig_temp);
  }
};

class UpperCaseDecoder : public CaseEncoder {
private:
  std::unique_ptr<std::string> buffer_;
  absl::string_view input_;

  int state_ = 0;
  bool allUp_{false};

public:
  UpperCaseDecoder() {}

  std::pair<absl::string_view, int> normalizePrefix(absl::string_view input) {
    if(!buffer_) {
      buffer_.reset(new std::string(input.data(), input.size()));
      input_ = absl::string_view(*buffer_);
    }

    if(input_[0] == cAllUppercase) {
      const_cast<char&>(input_[0]) = cUppercase;
      allUp_ = true;
    } else if (input_[0] == cTitlecase) {
      allUp_ = false;
    } else if (input_[0] == cLowercase) {
      allUp_ = false;
    }

    auto p = CaseEncoder::normalizePrefix(input_);
    int consumed = p.second;

    if(input_[0] == cUppercase) {
      if(state_ == 0) { 
        input_.remove_prefix(consumed - 1);
        const_cast<char&>(input_[0]) = cUppercase;
        state_ = 1;
      } else if(state_ == 1) {
        if(consumed > 1) {
          input_.remove_prefix(consumed - 1);
          const_cast<char&>(input_[0]) = cUppercase;
          p.second = consumed - 1;
          state_ = 1;
        } else {
          input_.remove_prefix(consumed);
          p.first.remove_prefix(1);
          p.second = 0;
          state_ = 0;
        }
      }
    } else if(input_[0] == cLowercase) {
      input_.remove_prefix(consumed);
      p.first.remove_prefix(1);
      state_ = 0;
    } else {
      if(allUp_) {
        p.first = absl::string_view(input.data(), p.first.size());
        input_.remove_prefix(consumed - 1);
        const_cast<char&>(input_[0]) = cUppercase;
        state_ = 1;
      } else {
        input_.remove_prefix(consumed);
        state_ = 0;
      }
    }

    return p;
  }
};

}  // namespace normalizer
}  // namespace sentencepiece
#endif  // NORMALIZER_CASE_ENCODER_H_
