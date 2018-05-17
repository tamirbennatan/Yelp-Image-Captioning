import numpy as np
import copy

class SequenceCandidate(object):
    
    @staticmethod
    def template_seq(start_idx = 1, max_length = 15):
        seq = np.repeat(0,15)
        seq[0] = start_idx
        # start with a prob of 1
        probsum = 1
        return SequenceCandidate(seq, probsum, max_length)
        
        
    
    def __init__(self, seq, probsum, max_length = 15):
        assert len(seq) == max_length
        self._max_length = max_length
        self._seq = seq
        self._probsum = probsum
        # store the number of populated elements in sequence
        self._num_elem = max_length # temporarily assume sequence is fully populated. 
        for i in range(len(seq)):
            if seq[i] == 0:
                self._num_elem = i # update number of elements in the sequence. 
                break
    
    # returns a new candidate, with the new token added
    def add_token(self, token, prob):
        # see that there's room to add in the sequence
        if self._num_elem >= self._max_length:
            raise IndexError("Sequence is already populated.\nCan't add any more tokens to it.")
        # get a copy of the new candidate
        newcandidate = copy.deepcopy(self)
        # add the token to the sequence
        newcandidate._seq[self._num_elem] = token
        # increment the number of stored elements
        newcandidate._num_elem += 1
        # update the probability sum
        newcandidate._probsum += prob
        return(newcandidate)
        
    # get the average probability
    def _average_prob(self):
        return self._probsum/self._num_elem
    
    # it is assumed that the first word is the start token <startseq>, so it is ignored. 
    def to_words(self,reverse_tokenizer, end_idx):
        # build up the words you want to output
        out_words = []
        for idx in self._seq[1:]:
            if idx == 0 or idx == end_idx:
                break
            out_words.append(reverse_tokenizer[idx])
        out_string = " ".join(out_words)
        return out_string
    
    # make the object sortabe **by the sum of the probability**
    def __lt__(self, other):
        try:
            return self._probsum < other._probsum
        except AttributeError: # don't know how to compare with general objects
            return NotImplemented