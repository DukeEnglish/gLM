#include "btree.hh"
#include "tokenizer.hh"

void addToTrie(B_tree * root_trie, processed_line ngram, unsigned int max_order, unsigned int max_node_size) {
    B_tree * next_level = root_trie;

    //Find the appropriate Btree for insertion. We only insert the last element from ngrams
    //As the previous ones would have been inserted according to the definition of the ngram language model
    for (unsigned int i = 0; i < ngram.ngrams.size() - 1; i++) {
        std::pair<B_tree_node *, int> result = next_level->find_element(ngram.ngrams[i]);
        next_level = result.first->words[result.second].next_level; //Get the next level btree node.
    }

    Entry to_insert;
    if (max_order != ngram.ngram_size) {
        //Create a btree that is going to host the future level.
        B_tree * future_level = new B_tree(max_node_size);

        //Now populate the entry
        to_insert.value = ngram.ngrams.back();
        to_insert.next_level = future_level;
        to_insert.prob = ngram.score;
        to_insert.backoff = ngram.backoff;  //Careful about the <unk> case
    } else {
        //We are at highest order ngram of the model. No next level and no backoff
        to_insert.value = ngram.ngrams.back();
        to_insert.next_level = nullptr;
        to_insert.prob = ngram.score;
        to_insert.backoff = 0.0;
    }

    //Now insert the entry
    next_level->insert_entry(to_insert);
}

size_t calculateTrieSize(B_tree * root_trie) {
    //Returns the total trie size in bytes.
    size_t ret = 0;
    std::queue<B_tree *> btrees_to_explore;
    btrees_to_explore.push(root_trie);

    while(!btrees_to_explore.empty()) {
        B_tree * current = btrees_to_explore.front();
        ret += current->getTotalTreeSize();
        btrees_to_explore.pop(); //We have processed the element, pop it.

        //Now add future elements to the queue by traversing the btree
        Pseudo_btree_iterator * iter = new Pseudo_btree_iterator(current->root_node);
        do {
            Entry * entry = iter->get_entry();
            if (entry->next_level) {
                btrees_to_explore.push(entry->next_level);
            }
            iter->increment();
        } while (!iter->finished);
        delete iter;
    }
    return ret;
}

//Compresses all of the btree os the tries
void compressTrie(B_tree * root_trie) {
    std::queue<B_tree *> btrees_to_explore;
    btrees_to_explore.push(root_trie);

    while(!btrees_to_explore.empty()) {
        B_tree * current = btrees_to_explore.front();
        btrees_to_explore.pop(); //We have processed the element, pop it.

        //Now add future elements to the queue by traversing the btree
        Pseudo_btree_iterator * iter = new Pseudo_btree_iterator(current->root_node);
        do {
            Entry * entry = iter->get_entry();
            if (entry->next_level) { //The second check shouldn't be necessary, except for unk! Investigate!
                if (entry->next_level->size == 0) {
                    //Purge empty btrees. Not sure how we get them that's happening though... Maybe because of UNK?
                    delete entry->next_level;
                    entry->next_level = nullptr;
                } else {
                    btrees_to_explore.push(entry->next_level);
                }
            }
            iter->increment();
        } while (!iter->finished);
        delete iter;
        current->compress();
    }
}

//Clears the Trie from memory.
void deleteTrie(B_tree * root_trie) {
    std::queue<B_tree *> btrees_to_explore;
    btrees_to_explore.push(root_trie);

    while(!btrees_to_explore.empty()) {
        B_tree * current = btrees_to_explore.front();
        btrees_to_explore.pop(); //We have processed the element, pop it.

        //Now add future elements to the queue by traversing the btree
        Pseudo_btree_iterator * iter = new Pseudo_btree_iterator(current->root_node);
        do {
            if (current->size == 0) {
                break; //Btrees with size of 0 don't span other btrees
            }
            Entry * entry = iter->get_entry();
            if (entry->next_level) { //The second check shouldn't be necessary, except for unk! Investigate!
                btrees_to_explore.push(entry->next_level);
            }
            iter->increment();
        } while (!iter->finished);
        delete iter;
        delete current;
    }
}

std::pair<Entry, unsigned short> findNgram(B_tree * root_trie, std::vector<unsigned int> ngrams) {
    //Returns the Entry and the order of the model found
    //We expect given an ngram a b c d to search for P(d|a b c) and the input vector should be [d, c, b, a]
    //For testing correct behaviour on the CPU
    B_tree * next_level = root_trie;
    Entry ret;
    unsigned short level = 0; //We couldn't even find the  first ngram

    for (auto vocabID : ngrams) {
        std::pair<B_tree_node *, int> result = next_level->find_element(vocabID);
        //Check if we have indeed found that element)
        if (result.first->words[result.second] == vocabID) {
            //We have indeed found it
            next_level = result.first->words[result.second].next_level;
            level++;
            ret = result.first->words[result.second];
        } else {
            break; //We didn't find anything, return the last found backoff
        }
    }

    return std::pair<Entry, unsigned short>(ret, level);
}

void trieToByteArray(std::vector<unsigned char>& byte_arr, B_tree * root_trie){
    bool pointer2Index = true; //We are converting the B_tree * to offset index.
    size_t offset = root_trie->getTotalTreeSize(pointer2Index); //Offset from the start of the the array to the desired element
                                                   //It is size_t to silence compiler warning, but maximum value should be the one permited by unsigned int
    std::queue<B_tree *> btrees_to_explore;
    btrees_to_explore.push(root_trie);

    while (!btrees_to_explore.empty()) {
        B_tree * current_level = btrees_to_explore.front();

        //Now add future elements to the queue by traversing the btree
        Pseudo_btree_iterator * iter = new Pseudo_btree_iterator(current_level->root_node);
        do {
            Entry * entry = iter->get_entry();
            if (entry->next_level) { //The second check shouldn't be necessary, except for unk! Investigate!
                entry->next_level = (B_tree *)offset; //This is no longer a pointer but an offset from beginning of array.
                offset+= entry->next_level->getTotalTreeSize(pointer2Index);
                btrees_to_explore.push(entry->next_level);
            } else {
                entry->next_level = 0; //When we don't have a child it's offset is 0
            }
            iter->increment();
        } while (!iter->finished);
        delete iter;

        //Convert the trie level to byte array
        current_level->toByteArray(byte_arr, pointer2Index);

        btrees_to_explore.pop();
    }
}

std::pair<bool, std::string> test_trie(const char * infile, unsigned short btree_node_size) {
    //Constructs a trie from an infile and then checks if it can find every token.
    ArpaReader pesho(infile);
    processed_line text = pesho.readline();
    B_tree * root_btree = new B_tree(btree_node_size);

    while (!text.filefinished){
        addToTrie(root_btree, text, pesho.max_ngrams, btree_node_size);
        text = pesho.readline();
    }

    //Btree constructed. Compress it. This is necessary to get rid of empty btrees.
    compressTrie(root_btree);

    ArpaReader pesho2(infile);
    processed_line text2 = pesho2.readline();
    bool correct = true;
    std::stringstream error;

    while (!text2.filefinished && correct) {
        std::pair<Entry, unsigned short> found = findNgram(root_btree, text2.ngrams);

        if (found.second) {
            correct = found.first.prob == text2.score;
            correct = correct && (found.first.backoff == text2.backoff);
        } else {
            error << "Ngram not found! " << text2 << std::endl;
            break;
        }
        if (!correct) {
            error << text2 << std::endl;
            error << "Ngram size is: " << text2.ngram_size << std::endl;
            error << "There has been an error! Score: Expected " << text2.score
            << " Got: " << found.first.prob << " Backoff expected: " << text2.backoff << " Got: " << found.first.backoff << std::endl;
            break;
        }
        text2 = pesho2.readline();
    }

    //Free all the used memory
    deleteTrie(root_btree);

    //Now search for every single entry.
    return std::pair<bool, std::string>(correct, error.str());
}
