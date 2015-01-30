#include "structs.h"
#include <utility>
#include <iostream>
#include <cstdlib>

class B_tree_node {
    public:
        unsigned short num_children; //Keep track of number of children. Will be removed in the GPU version.
        unsigned short num_keys;
        unsigned short max_keys;
        Entry * keys;
        bool is_leaf; //Maybe could replace with a a check for NULL pointer below
        B_tree_node ** children;
        B_tree_node * parent; //Backpointer to parent node

        //Operations
        B_tree_node(unsigned short);
        ~B_tree_node();
        void insert_entry(Entry new_entry, unsigned short position); //Inserting in the keys array. Should be called only if there is space.
        void split();
        void insert_child(B_tree_node * new_child, unsigned short position);
};

B_tree_node::B_tree_node(unsigned short min_degree) {
    num_keys = 0;
    keys = nullptr;// new Entry[2*min_degree - 1]; //Maximum number of entries;
    parent = nullptr;
    is_leaf = false;
    num_children = 2*min_degree;
    children = new B_tree_node *[num_children]; //Maximum number of child nodes
    max_keys = 2*min_degree - 1;
}

B_tree_node::~B_tree_node(){
    delete[] keys;
    //Delete all child nodes, effectively reccursively calling this function.
    for (int i = 0; i<num_children; i++){
        if (children[i]) {
            delete children[i];
        } else {
            break;
        }
    }
    delete[] children;
}


class B_tree{
    public:
        unsigned short minimum_degree;    
        B_tree_node * root_node;
        B_tree(unsigned short);
        ~B_tree(); //Destructor.
        void insert_Entry(Entry& new_entry);
        std::pair<B_tree_node *, unsigned short> findPosition(unsigned short value);
        void draw_tree();
};

B_tree::B_tree (unsigned short min_degree){
    minimum_degree = min_degree;
    B_tree_node * root = new B_tree_node(min_degree);
    root_node = root;
}

std::pair<B_tree_node *, unsigned short> B_tree::findPosition(unsigned short value) {
    //Get a pointer to the node where the value should be inserted
    B_tree_node * current_location = root_node;
    unsigned short position;

    while (true) {
        //Linear search for position, fast enough as it is only for construction.
        //We loop until we find that our value is greater than the one currently
        //examined which means that it should go on the right of the last previously
        //smaller value (or in the child after that)

        for (unsigned short i = 0; i<root_node->num_keys; i++){
            if (value < root_node->keys[i].value) {
                position = i;
            } else {
                break;
            }
        }

        if (!root_node->children[0]) {
            break;
            //Break if we are at the last node.
        } else {
            if (!current_location->children[position]) {
                std::cout << "We have a null pointer at position " << position << " exiting." << std::endl;
                std::exit(EXIT_FAILURE);
            }
            current_location = current_location->children[position];
        }

    }

    std::pair <B_tree_node *, unsigned short> retval (current_location, position); 
    return retval;
}

void B_tree::insert_Entry (Entry& new_entry) {
    std::pair<B_tree_node *, unsigned short> location = findPosition(new_entry.value);
    location.first->insert_entry(new_entry, location.second);

    if (location.first -> num_keys > location.first->max_keys) {
        location.first->split();
    }


}

void B_tree_node::insert_entry (Entry new_entry, unsigned short position) {
    //Create a new array and deallocate the old one including the new entry inside;
    Entry * new_arr = new Entry[num_keys + 1];

    if (num_keys == 0) {
        new_arr[0] = new_entry;
    } else {

        for (unsigned short i = 0; i<num_keys; i++) {
            if (i < position) {
                new_arr[i] = keys[i];
            } else if (i == position) {
                new_arr[i] = new_entry;
                new_arr[i+1] = keys[i];
            } else {
                new_arr[i+1] = keys[i];
            }
        }
    }

    delete[] keys; //Free the memory taken by the old array
    keys = new_arr;
    num_keys++;
}

void B_tree_node::insert_child (B_tree_node * new_child, unsigned short position) {
    B_tree_node ** children_new = new B_tree_node*[num_children +1];

    if (num_children == 0) {
        children_new[0] = new_child;
    } else {

        for (unsigned short i =0; i < num_children; i++) {
            if (i < position) {
                children_new[i] = children[i];
            } else if (i == position) {
                children_new[i] = new_child;
                children_new[i+1] = children[i];
            } else {
                children_new[i+1] = children[i];
            }
        }
    }

    delete[] children;
    children = children_new;
    num_children++;
}

void B_tree_node::split(){
    //Integer division to get middle index
    unsigned short mid_idx = num_keys/2;
    Entry entry_to_move = keys[mid_idx];
    
    //Create new node
    B_tree_node * right_node = new B_tree_node(max_keys);

    //Transfer everything to it

    //Transfer keys.
    right_node->keys = new Entry[mid_idx]; //Divide array in 2;
    unsigned short j = 0; //Iterate over the keys array;
    for (unsigned short i = mid_idx+1; i<num_keys; i++) {
        right_node->keys[j] = keys[i];
        j++;
    }
    right_node->num_keys = mid_idx;

    //Transfer children
    j = 0;
    for (unsigned short i = mid_idx+1; i<num_children; i++) {
        right_node->children[j] = children[i];
        j++;
    }
    right_node->num_children = num_children/2;

    //Trim the left node to the previous size. Only need to change size, 
    //We don't care for slightly higher memory usage
    //WE DO CARE, need to deallocate it. Create new arrays;
    num_children = num_children - num_children/2; //account for odd sizes
    num_keys = num_keys - num_keys/2;


    //Create new sub arrays
    B_tree_node ** children_tmp = new B_tree_node *[num_children];
    Entry * keys_tmp = new Entry[num_keys];

    for (unsigned short i=0; i<num_children; i++) {
        children_tmp[i] = children[i];
    }

    for (unsigned short i=0; i<num_keys; i++) {
        keys_tmp[i] =keys[i];
    }

    //Free old arrays;
    delete[] keys;
    delete[] children;

    //Assign the new arrays;
    keys = keys_tmp;
    children = children_tmp;

    //Parent might be a nullptr here. We might need to add a new root node.
    if (parent) {
        right_node->parent = parent; //Set the parent. same as the current node.
        //Insert entry in parent and put a link to the new child
        parent->insert_entry(entry_to_move, parent->num_keys/2);
        parent->insert_child(right_node, parent->num_keys/2 + 1);
        //Check if splitting is necesssary.
        if (parent->num_keys > parent->max_keys) {
            parent->split();
        }

    } else {
        //We are a root node and we have run out of space. We need to create a new root.
        //In order to avoid reasigning a root node in the tree, create a new node and copy
        //the left side onto it
        B_tree_node * left_node = new B_tree_node(max_keys);
        left_node->children = children;
        left_node->keys = keys;
        left_node->parent = this;
        left_node->num_keys = num_keys;
        left_node->num_children = num_children;

        //Now make the node contain only the single key
        delete[] children;
        delete[] keys;
        num_children = 0;
        num_keys = 0;
        this->insert_entry(entry_to_move, 0);
        this->insert_child(left_node, 0);
        this->insert_child(right_node, 1);
    }


}

void print_node(B_tree_node * node) {
    std::cout << "Node_ID" << node << std::endl;
    std::cout << "Num keys: " << node->num_keys;
    std::cout << "Num children: " << node-> num_children;
    std::cout << "Values: ";
    for (unsigned short i = 0; i<node->num_keys; i++) {
        std::cout << node->keys[i].value << ' ';
    }
    for (unsigned short i = 0; i<node->num_children; i++) {
        print_node(node->children[i]);
    }

}

void B_tree::draw_tree() {
    //Draws what is currently in the tree. For debugging purposes
    print_node(root_node);
}

B_tree::~B_tree(){
    delete root_node;
}

