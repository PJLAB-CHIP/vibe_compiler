
#ifndef _TLV_BOX_H_
#define _TLV_BOX_H_

#include "key_list.h"


typedef struct _tlv_box {
    key_list_t *m_list;
    uint8_t  scope;
    uint8_t  m_serialized_buffer_is_ref;//buffer 是否引用的flag
    int m_serialized_bytes;
    const unsigned char *m_serialized_buffer;
} tlv_box_t;


tlv_box_t *tlv_box_create(uint8_t scope);//scope是内存的范围目前有RTOS_SCOPE,SPM_SCOPE,LOCAL_TILE_SCOPE
int tlv_box_destroy(tlv_box_t *box);

// 把存储在 tlv_bot_t中的 tlv 的 item 序列化到数组中
int tlv_box_serialize(tlv_box_t *box);

// tlv结构的数据在 buffer 中, 复制 buffer 到新申请的内存中,然后在新地址上解析,把解析结构保存到 tlv_box中
tlv_box_t *tlv_box_parse_and_copy_buffer(const unsigned char *buffer, uint32_t buffersize, uint8_t scope);

// tlv结构的数据在 buffer 中, 在 buffer地址上解析,把解析结构保存到 tlv_box中, tlv_box使用过程中须在 buffer的生命周期中使用
tlv_box_t *tlv_box_parse_and_ref_buffer(const unsigned char *buffer, uint32_t buffersize, uint8_t scope);


const unsigned char* tlv_box_get_buffer(tlv_box_t* box);
int tlv_box_get_size(tlv_box_t *box);

// 用来计算如果 value 在相应类型情况下 T + L + V 需要多少字节来存储
int get_size_if_put_char();
int get_size_if_put_short();
int get_size_if_put_int();
int get_size_if_put_long();
int get_size_if_put_longlong();
int get_size_if_put_float();
int get_size_if_put_double();
int get_size_if_put_string(const char* value);
int get_size_if_put_bytes(int bytes_length);




// 这些 tlv_box_put_*** 开头的 api 是用来构建 多个 tlv 序列用的
int tlv_box_put_char(tlv_box_t *box,uint32_t type,char value);
int tlv_box_put_short(tlv_box_t *box,uint32_t type,short value);
int tlv_box_put_int(tlv_box_t *box,uint32_t type,int value);
int tlv_box_put_long(tlv_box_t *box,uint32_t type,long value);
int tlv_box_put_longlong(tlv_box_t *box,uint32_t type,long long value);
int tlv_box_put_float(tlv_box_t *box,uint32_t type,float value);
int tlv_box_put_double(tlv_box_t *box,uint32_t type,double value);

//把 string相关的 tlv 的 item 放到 tlv序列中,string的值被拷贝到新申请的数组上
int tlv_box_put_string_and_copy_it(tlv_box_t *box,uint32_t type,const char* value);

//把 byte数组相关的 tlv 的 item 放到 tlv序列中, byte数组的值被拷贝到新申请的数组上
int tlv_box_put_bytes_and_copy_it(tlv_box_t *box,uint32_t type,const unsigned char *value,uint32_t length);

//把 box object相关的 tlv 的 item 放到 tlv序列中, box object的值被拷贝到新申请的数组上
int tlv_box_put_object_and_copy_it(tlv_box_t *box,uint32_t type,tlv_box_t *object);

//把 string相关的 tlv 的 item 放到 tlv序列中,string被引用, 使用该tlv item 须在 string的生命周期中使用
int tlv_box_put_string_and_ref_it(tlv_box_t *box,uint32_t type,const char* value);

//把 byte数组相关的 tlv 的 item 放到 tlv序列中, byte数组被引用,使用该tlv item 须在 byte数组的生命周期中使用
int tlv_box_put_bytes_and_ref_it(tlv_box_t *box,uint32_t type,const unsigned char *value,uint32_t length);

//把 box object相关的 tlv 的 item 放到 tlv序列中, box object被引用,使用该tlv item 须在 box object的生命周期中使用
int tlv_box_put_object_and_ref_it(tlv_box_t *box,uint32_t type,tlv_box_t *object);



//  有多个tlv需要查找的情况下使用
int tlv_box_get_char(tlv_box_t *box,uint32_t type,char *value);
int tlv_box_get_short(tlv_box_t *box,uint32_t type,short *value);
int tlv_box_get_int(tlv_box_t *box,uint32_t type,int *value);
int tlv_box_get_long(tlv_box_t *box,uint32_t type,long *value);
int tlv_box_get_longlong(tlv_box_t *box,uint32_t type,long long *value);
int tlv_box_get_float(tlv_box_t *box,uint32_t type,float *value);
int tlv_box_get_double(tlv_box_t *box,uint32_t type,double *value);

// 这三个 api需要提前申请内存，传进去，这个适用于知道 box中 tlv的具体情况下的，如果不知道可以用tlv_box_get_bytes_ptr得到 tlv的
// 的 value和 length，然后根据实际情况操作，比如拷贝之类的
int tlv_box_get_string(tlv_box_t *box,uint32_t type,char *value,uint32_t* length);
int tlv_box_get_bytes(tlv_box_t *box,uint32_t type,unsigned char *value,uint32_t* length);

// 这个适用于 tlv中的 value 的内容是 tlv的结构的情况, 这个 api会拷贝 value的值到新申请的内存地址
int tlv_box_get_object_heavy_version(tlv_box_t *box,uint32_t type,tlv_box_t **object);

//可以得到 tlv的的 value 的地址 和 length,然后用户更加实际情况操作,比如访问,拷贝或者强转为相应的结构体对象等
int tlv_box_get_bytes_ptr(tlv_box_t *box,uint32_t type,const unsigned char **value,uint32_t* length);

// 这个适用于 tlv中的 value 的内容是 tlv的结构的情况, 这个 api会 引用value的地址，所以需要注意value生命周期的问题
int tlv_box_get_object(tlv_box_t *box,uint32_t type,tlv_box_t **object);




//  这些版本是不需要tlv_box_t的, 因为 tlv_box_t结构体使用过程需要 malloc一些辅助的数据结构,会有些性能损耗
//  这些版本也有局限就是每次都要从头遍历,适用于不需要从 tlv中获取一个item的情况
int tlv_get_char(const unsigned char *buffer,const  uint32_t buffersize,uint32_t type,char *value);
int tlv_get_short(const unsigned char *buffer,const  uint32_t buffersize,uint32_t type,short *value);
int tlv_get_int(const unsigned char *buffer,const  uint32_t buffersize,uint32_t type,int *value);
int tlv_get_long(const unsigned char *buffer,const  uint32_t buffersize,uint32_t type,long *value);
int tlv_get_longlong(const unsigned char *buffer,const  uint32_t buffersize,uint32_t type,long long *value);
int tlv_get_float(const unsigned char *buffer,const  uint32_t buffersize,uint32_t type,float *value);
int tlv_get_double(const unsigned char *buffer,const  uint32_t buffersize,uint32_t type,double *value);

// 这2个api需要提前申请内存，传进去，这个适用于知道 buffer 中 tlv的具体情况下的，会有拷贝的动作
int tlv_get_string(const unsigned char *buffer,const  uint32_t buffersize,uint32_t type,char *value,uint32_t* length);
int tlv_get_bytes(const unsigned char *buffer,const  uint32_t buffersize,uint32_t type,unsigned char *value,uint32_t* length);

//用来得到tlv的 中 某个 tlv 的buffer的 引用 和 length
int tlv_get_object(const unsigned char *buffer,const  uint32_t buffersize,uint32_t type,
    const unsigned char **object_buffer_value,uint32_t* object_buffer_value_length);

//用来得到tlv的的 value 的引用 和 length  ，然后根据实际情况操作，比如拷贝之类的
int tlv_get_bytes_ptr(const unsigned char *buffer,const  uint32_t buffersize,uint32_t type,const unsigned char **value,uint32_t* length);


#endif //_TLV_BOX_H_
