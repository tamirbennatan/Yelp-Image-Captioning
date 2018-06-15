import numpy as np

from sequence_candidate import SequenceCandidate


def generate_predictions_beam(img_id, features, caption_model, reverse_tokenizer, width, num_neighbors,
                              top_n = 3, end_idx = 2, max_length = 15, ignore_idx = [4,61,345], alpha = .6):
    # isolate the photo features
    photo_features = features[img_id]
    # keep track of the accepted sequences
    accepted_sequences = []
    # keep track of the current population
    population = []
    # add a start sequence to the population
    start_sequence = SequenceCandidate.template_seq(ignore_idx = ignore_idx, alpha = alpha)
    population.append(start_sequence)
    for i in range(max_length - 1):
        tmp = []
        for cand_seq in population:
            # pdb.set_trace()
            pred = caption_model.predict([photo_features, cand_seq._seq.reshape(1,-1)], verbose=0)[0]
            # sort the predicted next words by their probabilities
            pred_argsort = pred.argsort()
            # add candidates for each of the <num_neighbors> neighbors
            for next_idx in pred_argsort[-num_neighbors:]:
                # if we're starting to repeat bigrams, accept the current candidate
                if (cand_seq.final_token(), next_idx) in cand_seq._bigrams:
                    accepted_sequences.append(cand_seq)
                    continue
                # add the predicted word to get a new candidate
                next_prob = pred[next_idx]
                new_candidate = cand_seq.add_token(next_idx,next_prob)
                # if the next suggested token is <endseq>, add to accepted_sequences
                if next_idx == end_idx:
                    accepted_sequences.append(new_candidate)
                else:
                    tmp.append(new_candidate)
        # prune the population to keep only the top <width> candidates.
        try:
            population = sorted(tmp)[-width:]
        except:
            # fewer than <width> individuals remain - stop growing tree and keep curren partial sequences
            population = tmp
            break

    accepted_sequences = sorted(accepted_sequences + population, reverse = True)
    # build output JSON data
    num_accepted = 0
    values = []
    probs = []
    strings = []
    for acc_seq in accepted_sequences:
        # convert current sequence to words
        seq_string = acc_seq.to_words(reverse_tokenizer,end_idx)
        # if its not already in one of the word lists, accept it.
        if seq_string not in strings:
            strings.append(seq_string)
            probs.append(acc_seq.probsum())
            num_accepted += 1
            # if you've already accepted <top_n>, you're done
            if num_accepted >= top_n:
                break
    # return the strings and the probabilities
    output = list(zip(strings,probs))
    return output
