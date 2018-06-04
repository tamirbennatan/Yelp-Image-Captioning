import numpy as np
import copy

class SequenceCandidate(object):
    
    @staticmethod
    def template_seq(start_idx = 1, max_length = 15, ignore_idx = None, alpha = .9):
        seq = np.repeat(0,15)
        seq[0] = start_idx
        # also keep track of the probabilities you've seen
        probs = np.repeat(0.0,15)
        probs[0] = 1
        return SequenceCandidate(seq, probs, max_length, ignore_idx, alpha)
        
        
    
    def __init__(self, seq, probs, max_length = 15, ignore_idx = None, alpha = .9):
        assert len(seq) == max_length
        self._max_length = max_length
        self._seq = seq
        self._probs = probs
        # store the number of populated elements in sequence
        self._num_elem = max_length # temporarily assume sequence is fully populated. 
        for i in range(len(seq)):
            if seq[i] == 0:
                self._num_elem = i # update number of elements in the sequence. 
                break
        # keep track of which bigrams are in the sequence
        self._bigrams = set()
        self._ignore_idx = ignore_idx
        if ignore_idx is None:
            self._ignore_idx = []
        self._prob_weights = [alpha**i for i in range(max_length)]
    
    # returns a new candidate, with the new token added
    def add_token(self, token, prob):
        # see that there's room to add in the sequence
        if self._num_elem >= self._max_length:
            raise IndexError("Sequence is already populated.\nCan't add any more tokens to it.")
        # get a copy of the new candidate
        newcandidate = copy.deepcopy(self)
        # add the token to the sequence
        newcandidate._seq[self._num_elem] = token
        # update the probability sum
        newcandidate._probs[self._num_elem] = prob
        # add the newly added bigram to the list of bigrams
        newcandidate._bigrams.add(tuple(newcandidate._seq[self._num_elem - 1 : newcandidate._num_elem + 1]))
        # increment the number of stored elements
        newcandidate._num_elem += 1
        return(newcandidate)
    
    def probsum(self):
        # sum of the word probabilities, ignoring the indecies in `ignore_idx`.
        valid_probs = self._probs[~np.in1d(self._seq, self._ignore_idx)]
        # return a weighted sum
        return np.sum(np.multiply(valid_probs, self._prob_weights[:len(valid_probs)]))
    
    def final_token(self):
        return self._seq[self._num_elem - 1]
    
    # it is assumed that the first word is the start token <startseq>, so it is ignored. 
    def to_words(self,reverse_tokenizer, end_idx):
        # build up the words you want to output
        out_words = []
        for i in range(1,len(self._seq)):
            # current word index
            idx = self._seq[i]
            if idx == 0 or idx == end_idx:
                break
            if idx in self._ignore_idx:
                continue
            # don't add repeated words
            if self._seq[i - 1] != idx:
                out_words.append(reverse_tokenizer[idx])
        out_string = " ".join(out_words)
        return out_string
    
    # make the object sortabe **by the sum of the probability**
    def __lt__(self, other):
        try:
            return self.probsum() < other.probsum()
        except AttributeError: # don't know how to compare with general objects
            return NotImplemented