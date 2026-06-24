
void rt_system_heap_init_with_scope_ex(rt_size_t begin_addr,rt_size_t size, rt_uint8_t scope);

void rt_system_heap_init_with_scope(void *begin_addr, void *end_addr, rt_uint8_t scope);
void *rt_malloc_with_scope(rt_size_t size, rt_uint8_t scope);
void *rt_realloc_with_scope(void *ptr, rt_size_t newsize, rt_uint8_t scope);
void *rt_calloc_with_scope(rt_size_t count, rt_size_t size, rt_uint8_t scope);
void rt_free_with_scope(void *ptr, rt_uint8_t scope);
void rt_memory_info_with_scope(rt_size_t *total,
                            rt_size_t *used,
                            rt_size_t *max_used,rt_uint8_t scope);
