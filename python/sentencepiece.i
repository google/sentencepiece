%module sentencepiece

# Python wrapper is generated with:
# % swig -python -c++ sentencepiece.i

%{
#include <sentencepiece_processor.h>
#if PY_VERSION_HEX >= 0x03000000
#undef PyString_Check
#define PyString_Check(name) PyUnicode_Check(name)
#define PyString_AsStringAndSize(obj, s, len) {*s = PyUnicode_AsUTF8AndSize(obj, len);}
#define PyString_FromStringAndSize(s, len) PyUnicode_FromStringAndSize(s, len)
#endif
%}

%ignore sentencepiece::SentencePieceProcessor::Encode(std::string const &, std::vector<std::string>*) const;
%ignore sentencepiece::SentencePieceProcessor::Encode(std::string const &, std::vector<int>*) const;
%ignore sentencepiece::SentencePieceProcessor::Encode(std::string const &, SentencePieceText *) const;
%ignore sentencepiece::SentencePieceProcessor::Decode(std::vector<std::string> const &,std::string *) const;
%ignore sentencepiece::SentencePieceProcessor::Decode(std::vector<int> const &, std::string *) const;
%ignore sentencepiece::SentencePieceProcessor::Decode(std::vector<std::string> const &, SentencePieceText *) const;
%ignore sentencepiece::SentencePieceProcessor::Decode(std::vector<int> const &, SentencePieceText *) const;
%ignore sentencepiece::SentencePieceProcessor::model_proto;
%ignore sentencepiece::SentencePieceProcessor::Load(std::istream *);
%ignore sentencepiece::SentencePieceProcessor::LoadOrDie(std::istream *);
%ignore sentencepiece::SentencePieceProcessor::model_proto();

%extend sentencepiece::SentencePieceProcessor {
  std::vector<std::string> Encode(const std::string& input) const {
    std::vector<std::string> output;
    $self->Encode(input, &output);
    return output;
  }

  std::vector<std::string> EncodeAsPieces(const std::string& input) const {
    std::vector<std::string> output;
    $self->Encode(input, &output);
    return output;
  }

  std::vector<int> EncodeAsIds(const std::string& input) const {
    std::vector<int> output;
    $self->Encode(input, &output);
    return output;
  }

  std::string Decode(const std::vector<std::string>& input) const {
    std::string output;
    $self->Decode(input, &output);
    return output;
  }

  std::string DecodePieces(const std::vector<std::string>& input) const {
    std::string output;
    $self->Decode(input, &output);
    return output;
  }

  std::string DecodeIds(const std::vector<int>& input) const {
    std::string output;
    $self->Decode(input, &output);
    return output;
  }

  int __len__() {
    return $self->GetPieceSize();
  }

  int __getitem__(const std::string& key) const {
    return $self->PieceToId(key);
  }
}

%typemap(out) std::vector<int> {
  $result = PyList_New($1.size());
  for (size_t i = 0; i < $1.size(); ++i)
    PyList_SetItem($result, i, PyInt_FromLong((long)$1[i]));
}

%typemap(out) std::vector<std::string> {
  $result = PyList_New($1.size());
  for (size_t i = 0; i < $1.size(); ++i)
    PyList_SetItem($result, i, PyString_FromStringAndSize($1[i].data(), $1[i].size()));
}

%typemap(out) std::string {
  $result = PyString_FromStringAndSize($1.data(), $1.size());
}

%typemap(in) const std::string & {
  std::string *out = nullptr;
  if (PyString_Check($input)) {
    char *str = nullptr;
    Py_ssize_t str_size = 0;
    PyString_AsStringAndSize($input, &str, &str_size);
    out = new std::string(str, str_size);
  } else {
    PyErr_SetString(PyExc_TypeError,"not a string");
    return NULL;
  }
  $1 = out;
}

%typemap(in) const std::vector<std::string>& {
  std::vector<std::string> *out = nullptr;
  if (PyList_Check($input)) {
    const size_t size = PyList_Size($input);
    out = new std::vector<std::string>(size);
    for (size_t i = 0; i < size; ++i) {
      PyObject *o = PyList_GetItem($input, i);
      if (PyString_Check(o)) {
        char *str = nullptr;
        Py_ssize_t str_size = 0;
        PyString_AsStringAndSize(o, &str, &str_size);
        (*out)[i] = std::string(str, static_cast<size_t>(str_size));
      } else {
        PyErr_SetString(PyExc_TypeError,"list must contain strings");
        return NULL;
      }
    }
  } else {
    PyErr_SetString(PyExc_TypeError,"not a list");
    return NULL;
  }
  $1 = out;
}

%typemap(in) const std::vector<int>& {
  std::vector<int> *out = nullptr;
  if (PyList_Check($input)) {
    const size_t size = PyList_Size($input);
    out = new std::vector<int>(size);
    for (size_t i = 0; i < size; ++i) {
      PyObject *o = PyList_GetItem($input, i);
      if (PyInt_Check(o)) {
        (*out)[i] = static_cast<int>(PyInt_AsLong(o));
      } else {
        PyErr_SetString(PyExc_TypeError,"list must contain integers");
        return NULL;
      }
    }
  } else {
    PyErr_SetString(PyExc_TypeError,"not a list");
    return NULL;
  }
  $1 = out;
}

%typemap(freearg) const std::string& {
  delete $1;
}

%typemap(freearg) const std::vector<std::string>& {
  delete $1;
}

%typemap(freearg) const std::vector<int>& {
  delete $1;
}

%include <sentencepiece_processor.h>
