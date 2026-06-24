/*
 * Copyright (C) 2017-2019 Alibaba Group Holding Limited
 */


/******************************************************************************
 * @file     csi_kernel.h
 * @brief    header file for kernel definition
 * @version  V1.0
 * @date     02. June 2017
 ******************************************************************************/

#ifndef _CSI_KERNEL_
#define _CSI_KERNEL_

#include <stdint.h>
#include <errno.h>
#include  "tx8_config.h"
#include "csi_core.h"
#ifdef  __cplusplus
extern "C"
{
#endif

/* =================================================================================== */
/*                         Enumerations, structures, defines                           */
/* =================================================================================== */

/// Status code values returned by CSI-kernel functions. 0 - success, negative represents error code ,see errno.h
typedef int32_t k_status_t;

/// Kernel scheduler state.
typedef enum {
    KSCHED_ST_INACTIVE         =  0,         ///< Inactive: The kernel is not ready yet. csi_kernel_init needs to be executed successfully.
    KSCHED_ST_READY            =  1,         ///< Ready: The kernel is not yet running. csi_kernel_start transfers the kernel to the running state.
    KSCHED_ST_RUNNING          =  2,         ///< Running: The kernel is initialized and running.
    KSCHED_ST_LOCKED           =  3,         ///< Locked: The kernel was locked with csi_kernel_sched_lock. The functions csi_kernel_sched_unlock or csi_kernel_sched_restore_lock unlocks it.
    KSCHED_ST_SUSPEND          =  4,         ///< Suspended: The kernel was suspended using csi_kernel_sched_suspend. The function csi_kernel_sched_resume returns to normal operation
    KSCHED_ST_ERROR            =  5          ///< Error: An error occurred.
} k_sched_stat_t;

/// task state.
typedef enum {
    KTASK_ST_INACTIVE          =  0,         ///< Inactive.
    KTASK_ST_READY             =  1,         ///< Ready.
    KTASK_ST_RUNNING           =  2,         ///< Running.
    KTASK_ST_BLOCKED           =  3,         ///< Blocked.
    KTASK_ST_TERMINATED        =  4,         ///< Terminated.
    KTASK_ST_ERROR             =  5          ///< Error: An error occurred.
} k_task_stat_t;

/// timer state.
typedef enum {
    KTIMER_ST_INACTIVE       = 0,          ///< not running
    KTIMER_ST_ACTIVE         = 1,          ///< running
} k_timer_stat_t;

/// Timer type.
typedef enum {
    KTIMER_TYPE_ONCE         = 0,          ///< One-shot timer.
    KTIMER_TYPE_PERIODIC     = 1           ///< Repeating timer.
} k_timer_type_t;

/// event option.
typedef enum {
    KEVENT_OPT_SET_ANY       = 0,          ///< Check any bit in flags to be 1.
    KEVENT_OPT_SET_ALL       = 1,          ///< Check all bits in flags to be 1.
    KEVENT_OPT_CLR_ANY       = 2,          ///< Check any bit in flags to be 0.
    KEVENT_OPT_CLR_ALL       = 3           ///< Check all bits in flags to be 0.
} k_event_opt_t;

/// Priority definition.
typedef uint8_t   k_priority_t ;
/// Entry point of a task.
typedef void (*k_task_entry_t)(void *arg);

/// Entry point of a timer call back function.
typedef void (*k_timer_cb_t)(void *arg);

/// \details Task handle identifies the task.
typedef void *k_task_handle_t;

/// \details Timer handle identifies the timer.
typedef void *k_timer_handle_t;

/// \details Event Flags handle identifies the event flags.
typedef void *k_event_handle_t;

/// \details Mutex handle identifies the mutex.
typedef void *k_mutex_handle_t;

/// \details Semaphore handle identifies the semaphore.
typedef void *k_sem_handle_t;

/// \details Memory Pool handle identifies the memory pool.
typedef void *k_mpool_handle_t;

/// \details Message Queue handle identifies the message queue.
typedef void *k_msgq_handle_t;



/* =================================================================================== */
/*                          Kernel Management Functions                                */
/* =================================================================================== */

/// Initialize the Kernel. Before it is successfully executed, no RTOS function should be called
/// \return execution status code. \ref k_status_t
k_status_t csi_kernel_init(void);

/// Start the kernel .It will not return to its calling function in case of success
/// \return execution status code. \ref k_status_t
k_status_t csi_kernel_start(void);




/* =================================================================================== */
/*                             Task Management Functions                               */
/* ====================================================================c=============== */

/// Create a task and add it to Active Tasks.
/// \param[in]     task          task function.
/// \param[in]     name          the name of task.
/// \param[in]     arg           pointer that is passed to the task function as start argument.
/// \param[in]     prio          task priority.
/// \param[in]     time_quanta   the amount of time (in clock ticks) for the time quanta when round robin is enabled,if Zero, then use FIFO sched
/// \param[in]     stack    stack base.
/// \param[in]     stack_size    stack size.
/// \param[in]     task_handle   reference to a task handle.
/// \return execution status code. \ref k_status_t
k_status_t csi_kernel_task_new(k_task_entry_t task, const char *name, void *arg,
                      k_priority_t prio, uint32_t time_quanta,
                      void *stack, uint32_t stack_size, k_task_handle_t *task_handle);


/// Create a task and add it to Active Tasks.
/// \param[in]     task          task function.
/// \param[in]     name          the name of task.
/// \param[in]     arg           pointer that is passed to the task function as start argument.
/// \param[in]     prio          task priority.
/// \param[in]     task_handle   reference to a task handle.
/// \cpuid[in]     cpuid          需要绑定的cpuid
/// \return execution status code. \ref k_status_t
k_status_t csi_kernel_create_task(k_task_entry_t task, const char *name, void *arg,
                               k_priority_t prio,uint32_t stack_size,
                               k_task_handle_t *task_handle,uint32_t cpuid);

/// Delete a task.
/// \param[in]     task_handle      task handle to operate.
/// \return execution status code. \ref k_status_t
k_status_t csi_kernel_task_del(k_task_handle_t task_handle);

/// Return the task handle of the current running task.
/// \return task handle for reference by other functions or NULL in case of error.
k_task_handle_t csi_kernel_task_get_cur(void);



/// Get current task state of a task.
/// \param[in]     task_handle     task handle to operate.
/// \return current task state of the specified task.
k_task_stat_t csi_kernel_task_get_stat(k_task_handle_t task_handle);

/// Change cur priority of current task.
/// \param[in]     priority      new priority value for the task function.
/// \return execution status code. \ref k_status_t
k_status_t csi_kernel_cur_task_set_prio(k_priority_t priority);

/// Change priority of a task.
/// \param[in]     task_handle     task handle to operate.
/// \param[in]     priority      new priority value for the task function.
/// \return execution status code. \ref k_status_t
k_status_t csi_kernel_task_set_prio(k_task_handle_t task_handle, k_priority_t priority);

/// Get current priority of a task.
/// \param[in]     task_handle     task handle to operate.
/// \return current priority value of the specified task.negative indicates error code.
k_priority_t csi_kernel_task_get_prio(k_task_handle_t task_handle);

k_priority_t csi_kernel_cur_task_get_prio();
/// Get name of a task.
/// \param[in]     task_handle     task handle to operate.
/// \return name of the task.
const char *csi_kernel_task_get_name(k_task_handle_t task_handle);

/// Suspend execution of a task.
/// \param[in]     task_handle     task handle to operate.
/// \return execution status code. \ref k_status_t
k_status_t csi_kernel_task_suspend(k_task_handle_t task_handle);

/// Resume execution of a task.
/// \param[in]     task_handle     task handle to operate.
/// \return execution status code. \ref k_status_t
k_status_t csi_kernel_task_resume(k_task_handle_t task_handle);

uint32_t csi_kernel_sched_suspend(void);

void csi_kernel_sched_resume(uint32_t sleep_ticks);

/// Terminate execution of a task.
/// \param[in]     task_handle     task handle to operate.
/// \return execution status code. \ref k_status_t
k_status_t csi_kernel_task_terminate(k_task_handle_t task_handle);

/// Exit from the calling task.
/// \return none
void csi_kernel_task_exit(void);

/// Pass control to next task that is in state \b READY.
/// \return execution status code. \ref k_status_t
k_status_t csi_kernel_task_yield(void);

/// Get number of active tasks.
/// \return number of active tasks.
uint32_t csi_kernel_task_get_count(void);

/// Get stack size of a task.
/// \param[in]     task_handle     task handle to operate.
/// \return stack size in bytes.
uint32_t csi_kernel_task_get_stack_size(k_task_handle_t task_handle);

/// Get available stack space of a thread based on stack watermark recording during execution.
/// \param[in]     task_handle     task handle to operate.
/// \return remaining stack space in bytes.
uint32_t csi_kernel_task_get_stack_space(k_task_handle_t task_handle);

/// Enumerate active tasks.
/// \param[out]    task_array    pointer to array for retrieving task handles.
/// \param[in]     array_items   maximum number of items in array for retrieving task handles.
/// \return number of enumerated tasks.
uint32_t csi_kernel_task_list(k_task_handle_t *task_array, uint32_t array_items);

/// System enter interrupt status.
/// \return execution status code. \ref k_status_t
k_status_t csi_kernel_intrpt_enter(void);

/// System exit interrupt status.
/// \return execution status code. \ref k_status_t
k_status_t csi_kernel_intrpt_exit(void);

/* =================================================================================== */
/*                                Generic time Functions                               */
/* =================================================================================== */

/// Waits for a time period specified in kernel ticks.
/// \param[in]     ticks        time ticks value
/// \return execution status code. \ref k_status_t
k_status_t csi_kernel_delay(uint32_t ticks);

/// Waits until an absolute time (specified in kernel ticks) is reached.
/// \param[in]     ticks         absolute time in ticks
/// \return execution status code. \ref k_status_t
k_status_t csi_kernel_delay_until(uint64_t ticks);

/// Convert kernel ticks to ms.
/// \param[in]     ticks     ticks which will be converted to ms
/// \return the ms of the ticks.
uint64_t csi_kernel_tick2ms(uint32_t ticks);

/// Convert ms to kernel ticks.
/// \param[in]     ms         ms which will be converted to ticks
/// \return the ticks of the ms.
uint64_t csi_kernel_ms2tick(uint32_t ms);

/// Waits for a time period specified in ms.
/// \param[in]     ms         time to be delayed in ms
/// \return execution status code. \ref k_status_t
k_status_t csi_kernel_delay_ms(uint32_t ms);

/// Get kernel ticks.
/// \return kernel ticks number
uint64_t csi_kernel_get_ticks(void);

/// Get the RTOS kernel tick frequency.
/// \return frequency of the kernel tick.
uint32_t csi_kernel_get_tick_freq(void);

/// Get the RTOS kernel system timer frequency.
/// \return frequency of the system timer.
uint32_t csi_kernel_get_systimer_freq(void);


/* =================================================================================== */
/*                              Event Management Functions                             */
/* =================================================================================== */

/// Create and Initialize an Event Flags object.
/// \return event flags handle for reference by other functions or NULL in case of error.
k_event_handle_t csi_kernel_event_new(void);

/// Delete an Event Flags object.
/// \param[in]     ev_handle     event flags handle to operate.
/// \return execution status code. \ref k_status_t
k_status_t csi_kernel_event_del(k_event_handle_t ev_handle);

/// Set the specified Event Flags.
/// \param[in]     ev_handle     event flags handle to operate.
/// \param[in]     flags         specifies the flags that shall be set.
/// \param[out]     ret_flags     The value of the event after setting.
/// \return execution status code. \ref k_status_t
k_status_t csi_kernel_event_set(k_event_handle_t ev_handle, uint32_t flags, uint32_t *ret_flags);

/// Clear the specified Event Flags.
/// \param[in]     ev_handle     event flags handle to operate.
/// \param[in]     flags         specifies the flags that shall be clear.
/// \param[out]     ret_flags     event flags before clearing.
/// \return execution status code. \ref k_status_t
k_status_t csi_kernel_event_clear(k_event_handle_t ev_handle, uint32_t flags, uint32_t *ret_flags);

/// Get the current Event Flags. This function allows the user to know “Who did it!”
/// \param[in]     ev_handle     event flags handle to operate.
/// \param[out]     ret_flags     The value of the current event.
/// \return execution status code. \ref k_status_t
k_status_t csi_kernel_event_get(k_event_handle_t ev_handle, uint32_t *ret_flags);

/// Wait for one or more Event Flags to become signaled.
/// \param[in]     ev_handle     event flags handle to operate.
/// \param[in]     flags         specifies the flags to wait for.
/// \param[in]     options       specifies flags options, \ref k_event_opt_t.
/// \param[in]     clr_on_exit   1 - event flags will be cleared before exit, otherwise event flags are not altered
/// \param[out]     actl_flags    The value of the event at the time either the bits being waited for became set, or the block time expired.
/// \param[in]     timeout       time out value in ticks if > 0, 0 in case of no time-out, negative in case of wait forever
/// \return execution status code. \ref k_status_t
k_status_t csi_kernel_event_wait(k_event_handle_t ev_handle, uint32_t flags,
                        k_event_opt_t options, uint8_t clr_on_exit,
                        uint32_t *actl_flags, int64_t timeout);


/* =================================================================================== */
/*                              Mutex Management Functions                             */
/* =================================================================================== */

/// Create and Initialize a Mutex object.
/// \return mutex handle for reference by other functions or NULL in case of error.
k_mutex_handle_t csi_kernel_mutex_new(void);

/// Delete a Mutex object.
/// \param[in]     mutex_handle      mutex handle to operate.
/// \return execution status code. \ref k_status_t
k_status_t csi_kernel_mutex_del(k_mutex_handle_t mutex_handle);

/// Acquire a Mutex or timeout if it is locked.
/// \param[in]     mutex_handle      mutex handle to operate.
/// \param[in]     timeout       time out value in ticks if > 0, 0 in case of no time-out, negative in case of wait forever
/// \return execution status code. \ref k_status_t
k_status_t csi_kernel_mutex_lock(k_mutex_handle_t mutex_handle, int64_t timeout);

/// Release a Mutex that was acquired by \ref csi_kernel_mutex_new.
/// \param[in]     mutex_handle      mutex handle to operate.
/// \return execution status code. \ref k_status_t
k_status_t csi_kernel_mutex_unlock(k_mutex_handle_t mutex_handle);

/// Get Thread which owns a Mutex object.
/// \param[in]     mutex_handle  mutex handle to operate.
/// \return task handle or NULL when mutex was not acquired.
k_task_handle_t csi_kernel_mutex_get_owner(k_mutex_handle_t mutex_handle);



/* =================================================================================== */
/*                            Semaphore Management Functions                           */
/* =================================================================================== */

/// Create and Initialize a Semaphore object.
/// \param[in]     max_count     maximum number of available tokens.
/// \param[in]     initial_count initial number of available tokens.
/// \return semaphore handle for reference by other functions or NULL in case of error.
k_sem_handle_t csi_kernel_sem_new(int32_t max_count, int32_t initial_count);

/// Delete a Semaphore object.
/// \param[in]     sem_handle  semaphore handle to operate.
/// \return execution status code. \ref k_status_t
k_status_t csi_kernel_sem_del(k_sem_handle_t sem_handle);

/// Acquire a Semaphore token or timeout if no tokens are available.
/// \param[in]     sem_handle  semaphore handle to operate.
/// \param[in]     timeout       time out value in ticks if > 0, 0 in case of no time-out, negative in case of wait forever
/// \return execution status code. \ref k_status_t
k_status_t csi_kernel_sem_wait(k_sem_handle_t sem_handle, int64_t timeout);


/// Acquire a Semaphore token or timeout if no tokens are available.
/// \param[in]     sem_handle  semaphore handle to operate.
/// \param[in]     timeout       time out value  in milliseconds. if > 0, 0 in case of no time-out, negative in case of wait forever
/// \return execution status code. \ref k_status_t
k_status_t csi_kernel_sem_wait_in_ms(k_sem_handle_t sem_handle, int64_t timeout);

/// Release a Semaphore token that was acquired by \ref csi_kernel_sem_wait.
/// \param[in]     sem_handle  semaphore handle to operate.
/// \return execution status code. \ref k_status_t
k_status_t csi_kernel_sem_post(k_sem_handle_t sem_handle);

/// Get current Semaphore token count.
/// \param[in]     sem_handle  semaphore handle to operate.
/// \return number of tokens available. negative indicates error code.
int32_t csi_kernel_sem_get_count(k_sem_handle_t sem_handle);


/* =================================================================================== */
/*                          Heap Management Functions                                  */
/* =================================================================================== */

/// Allocates size bytes and returns a pointer to the allocated memory.
/// \param[in]     scope     a scope to memory block.
/// \param[in]     size     Allocates size bytes.
/// \param[in]     caller   the function who call this interface or NULL.
/// \return  a pointer to the allocated memory.
void *csi_kernel_malloc(uint8_t scope,int32_t size, void *caller);

/// Frees the memory space pointed to by ptr
/// \param[in]     scope     a scope to memory block.
/// \param[in]     ptr      a pointer to memory block, return by csi_kernel_malloc or csi_kernel_realloc.
/// \param[in]     caller   the function who call this interface or NULL.
/// \return void
void csi_kernel_free(uint8_t scope,void *ptr, void *caller);

/// Changes the size of the memory block pointed to by ptr to size bytes
/// \param[in]     ptr      a pointer to memory block, return by csi_kernel_malloc or csi_kernel_realloc.
/// \param[in]     size     Allocates size bytes.
/// \param[in]     caller   the function who call this interface or NULL.
/// \return  a pointer to the allocated memory.
void *csi_kernel_realloc(void *ptr, int32_t size, void *caller);

/// Get csi memory used info.
/// \param[out]     total    the total memory can be use.
/// \param[out]     used     the used memory by malloc.
/// \param[out]     free     the free memory can be use.
/// \param[out]     peak     the peak memory used.
/// \return execution status code. \ref k_status_t.
k_status_t csi_kernel_get_mminfo(int32_t *total, int32_t *used, int32_t *free, int32_t *peak);

/// Dump csi memory .
/// \param void
/// \return execution status code. \ref k_status_t.
k_status_t csi_kernel_mm_dump(void);

#ifdef  __cplusplus
}
#endif

#endif  // _CSI_KERNEL_
