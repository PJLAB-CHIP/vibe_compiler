
#ifndef _TLV_KEY_LIST_H_
#define _TLV_KEY_LIST_H_

#include <stdbool.h>
#include <stdint.h>

typedef struct _value {
    void *value;
} value_t;

#define  RTOS_SCOPE 2
#define  SPM_SCOPE  1
#define  LOCAL_TILE_SCOPE 0 //DDR

typedef int tlv_key_t;
typedef void (*value_releaser)(value_t value,uint8_t scope);

#define key_compare(a, b) ((a==b)?1:0)

typedef struct key_list_node { 
    tlv_key_t key;
    value_t value;
    struct key_list_node *prev;
    struct key_list_node *next;
} key_list_node_t;

typedef struct key_list {
    int count;
    key_list_node_t *header;      
    value_releaser releaser;
    uint8_t  scope;
} key_list_t;

key_list_t *key_list_create(value_releaser releaser, uint8_t scope);
int key_list_destroy(key_list_t *list);

int key_list_count(key_list_t *list);
int key_list_keyset(key_list_t *list, tlv_key_t* array, int array_size);
int key_list_find_key(key_list_t *list, tlv_key_t key);

int key_list_add(key_list_t *list, tlv_key_t key, value_t value);
int key_list_get(key_list_t *list, tlv_key_t key, value_t *value);
int key_list_edit(key_list_t *list, tlv_key_t key, value_t value);
int key_list_delete(key_list_t *list, tlv_key_t key);

#define key_list_foreach(L,V) key_list_node_t *_node = NULL;\
    key_list_node_t* V;\
    for (V = _node = L->header; _node != NULL; V = _node = _node->next)

#endif
