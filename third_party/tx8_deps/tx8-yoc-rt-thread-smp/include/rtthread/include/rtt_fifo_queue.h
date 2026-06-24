#ifndef __RTT_FIFO_QUEUE_H__
#define __RTT_FIFO_QUEUE_H__

#include <rtthread.h>

// 定义队列节点（嵌入rt_list_node）
struct fifo_node {
    struct rt_list_node list;  // 链表节点
    void *data;               // 数据指针
};

// 定义FIFO队列结构
struct fifo_queue {
    struct rt_list_node head;  // 队列头节点
    rt_size_t count;           // 当前队列长度
    rt_size_t max_count;       // 最大队列长度
    rt_mutex_t mutex;          // 互斥锁（用于线程安全）
};

struct fifo_queue* fifo_create(rt_size_t max_count);
void fifo_delete(struct fifo_queue *queue);
rt_err_t fifo_push(struct fifo_queue *queue, void *data);
rt_err_t fifo_pop(struct fifo_queue *queue, void **data);
rt_err_t fifo_peek(struct fifo_queue *queue, void **data);
void fifo_clear(struct fifo_queue *queue);
rt_size_t fifo_count(struct fifo_queue *queue);
rt_bool_t fifo_is_empty(struct fifo_queue *queue);

void fifo_test();

#endif