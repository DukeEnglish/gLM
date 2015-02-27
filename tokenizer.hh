#include <iostream>
#include <string>
#include <boost/tokenizer.hpp>
#include <fstream>
#include <stdlib.h>
#include <map>

struct processed_line {
    unsigned short ngram_size;
    std::vector<unsigned int> ngrams;
    float score;
    float backoff;
    bool filefinished;
};

std::ostream& operator<< (std::ostream &out, processed_line &line) {
    out << "Number of ngrams: " << line.ngram_size << " End of file: " << line.filefinished << "\n" << line.score << ' ';
    for (int i = 0; i < line.ngram_size; i++) {
        out << line.ngrams[i] << ' ';
    }
    out << line.backoff << std::endl;
    return out;
}

class ArpaReader {
    private:
        std::ifstream arpafile;
        unsigned int vocabcounter; //Assigns vocabulary ids for words, starting for zero;
        unsigned short state; //0 for configuration, 1 for unigrams, 2 for bigrams, etc
        unsigned short max_ngrams;
        std::string next_ngrams_boundary; //The boundary condition for the next \N-grams string

        //Maps for converting to and from vocabulary ids to strings.
        std::map<unsigned int, std::string> decode_map;
        std::map<std::string, unsigned int> encode_map; 

    public:
        ArpaReader(const char *);
        processed_line readline();

};

ArpaReader::ArpaReader(const char * filename) {
    arpafile.open(filename);
    vocabcounter = 0;
    state = 0;

    //Read only the maximum ngrams for now, later on more things
    boost::char_separator<char> sep("=");

    std::string sLine = "";
    std::string unigrams = "\\1-grams:";
    while (sLine != unigrams){ //Check if this is how comparison actually works.
        boost::tokenizer<boost::char_separator<char> > tokens(sLine, sep);
        //We need the first token
        //Check for empty lines and discard them:
        if (tokens.begin() != tokens.end()) {
            //We are looking for the number right before the "=" sign.
            //For now let's restrict ourselves to models with up to 9grams
            //In the future the tokenizer would probably be rewritte.
            std::string first_token = *tokens.begin();
            max_ngrams = std::atoi(&first_token.back());
        }
        std::getline(arpafile, sLine); //Read in next line
    }
    state = 1; //We are now at unigrams
    next_ngrams_boundary = unigrams;
    next_ngrams_boundary.at(1) = std::to_string(state+1).c_str()[0];
};

processed_line ArpaReader::readline() {
    processed_line rettext;
    rettext.filefinished = false;
    //Check if we have reached the end of file.
    if (arpafile.eof()) {
        rettext.filefinished = true;
        arpafile.close();
        return rettext;
    }
    //Read in the line and tokenize it
    std::string sLine = "";
    std::getline(arpafile, sLine);
    boost::char_separator<char> sep("\t");
    boost::tokenizer<boost::char_separator<char> > tokens(sLine, sep);
    //Discard empty lines. If we encounter one we just go to the next one
    if (tokens.begin() == tokens.end()) {
        return this->readline();
    }

    if (sLine == next_ngrams_boundary) {
        //We have reached a new state. Update state information and return next line
        state++;
        next_ngrams_boundary.at(1) = std::to_string(state+1).c_str()[0];
        return this->readline();
    }
    rettext.ngram_size = state; //The state corresponds to the number of ngrams

    //Everything went normal, just unpack and tokenize the line
    boost::tokenizer<boost::char_separator<char> >::iterator it = tokens.begin();
    //The first token is the probability marker.
    rettext.score = stof(*it);
    it++;
    //The next (numerical value of) state tokens are ngrams
    for (int i = 0; i<state; i++){
        std::string current_word = *it;
        //Check if the current word has a vocabulary ID assigned and if not
        //produce one.
        std::map<std::string, unsigned int>::iterator id_found = encode_map.find(current_word);
        if (id_found != encode_map.end()) {
            //We already have that vocabulary id, just use it to append to the ngrams vector
            rettext.ngrams.push_back(id_found->second);
        } else {
            //We have an unseen ngram. assign the next vocabulary id
            encode_map.insert(std::pair<std::string, unsigned int>(current_word, vocabcounter));
            decode_map.insert(std::pair<unsigned int, std::string>(vocabcounter, current_word));
            vocabcounter++;
            rettext.ngrams.push_back(vocabcounter);
        }
        it++; //Go to the next token
    }

    //Now we either have end of line, or a backoff weight
    if (state != max_ngrams) {
        rettext.backoff = stof(*it);
    }
    //After assigning backoff weight, return the object
    return rettext;
}
