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

#include "case_encoder.h"

namespace sentencepiece {
namespace normalizer {

std::unique_ptr<CaseEncoder> CaseEncoder::Create(bool encodeCase, bool decodeCase, bool removeExtraWhiteSpace) {
  if(encodeCase && decodeCase) {
    LOG(ERROR) << "Cannot set both encodeCase=true and decodeCase=true";
    return nullptr;
  } else if(encodeCase) {
    return std::unique_ptr<CaseEncoder>(new UpperCaseEncoder(removeExtraWhiteSpace));
  } else if(decodeCase) {
    return std::unique_ptr<CaseEncoder>(new UpperCaseDecoder());
  } else {
    return nullptr;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////
// Implements finite state automaton for regex: Uu+(sss|p|$)+Uu+(sss|p|$)+(Uu+(sss|p|$)+)+
// std::regex seems to be causing weird issues with stack overflow etc. on Windows. 
// This here should just work.

constexpr size_t npos = -1; // invalid string position
constexpr int s = -1;       // sink state

// state transitions
constexpr int fsa[][4] = {
  {  7,  s,  s,  s},
  {  s,  4,  5,  1},
  {  3,  2, 14,  s},
  {  s,  s,  s,  1},
  {  3,  4,  5,  s},
  {  s,  s,  6,  s},
  {  s,  s,  4,  s},
  {  s,  s,  s,  8},
  {  s,  9, 10,  8},
  { 11,  9, 10,  s},
  {  s,  s, 12,  s},
  {  s,  s,  s, 13},
  {  s,  s,  9,  s},
  {  s,  2, 14, 13},
  {  s,  s, 15,  s},
  {  s,  s,  2,  s}
};

// mark end state
constexpr bool accept[16] = {
  false, false, false, false,
  true,  false, false, false,
  false, false, false, false,
  false, false, false, false
};

// map alphabet to position
inline int alphabet(char c) {
  switch (c) {
    case 'U': return 0;
    case 'p': return 1;
    case '$': return 1;
    case 's': return 2;
    case 'u': return 3;
    default: return -1; // invalid entry
  };
}

///////////////////////////////////////////////////////////////////////////////////////////

// state transition function
inline int delta(int state, char c) {
  int a = alphabet(c);
  return a != -1 ? fsa[state][a] : s;
}

// finds longest sequence that is accepted by the fsa starting 
// from the beginning up to given length
size_t searchLongestSuffix(const char* data, size_t length) {
  // init
  size_t found = npos;
  int state = 0;
  
  // is the start state an acceptor state?
  if(accept[state]) 
    found = 0;

  for(size_t i = 0; i < length; ++i) {
    // try state transition
    state = delta(state, data[i]);
    
    // if we ended up in a sink state, return what we found so far
    if(state == s)
      return found;

    // not a sink state, so check if it's an acceptor state and move pointer if yes
    if(accept[state])
      found = i + 1;
  }

  // we reached the end of the string, check if it makes us reach an acceptor state
  // in which case the whole sequence is matched
  state = delta(state, '$');
  if(state != s && accept[state])
      found = length;

  return found;
}

// find all the longest sequences that match the fsa in the string and return matched spans
// note, this is greedy and we restart search after the previous longest sequence
std::vector<std::pair<const char*, const char*>> search(const std::string& input) {
  std::vector<std::pair<const char*, const char*>> results;
  for(size_t i = 0; i < input.length(); ++i) {
    size_t found = searchLongestSuffix(input.data() + i, input.length() - i);
    if(found != npos) {
      results.emplace_back(input.data() + i, input.data() + i + found);
      i += found - 1;
    }
  }
  return results;
}

}
}