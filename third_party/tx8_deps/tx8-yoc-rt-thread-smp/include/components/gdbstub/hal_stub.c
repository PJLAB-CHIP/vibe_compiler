/*
 * I/O and interface portion of GDB stub
 *
 * File      : hal_stub.c
 * This file is part of RT-Thread RTOS
 * COPYRIGHT (C) 2006, RT-Thread Develop Team
 *
 * The license and distribution terms for this file may be
 * found in the file LICENSE in this distribution or at
 * http://www.rt-thread.org/license/LICENSE
 *
 * Change Logs:
 * Date           Author       Notes
 * 2014-07-04     Wzyy2      first version
 */
#include <rtthread.h>
#include <rthw.h>
#include "gdb_stub.h"

#if CONFIG_DEVICES_RVM_HAL
#include <devices/uart.h>
static rvm_dev_t *gdb_rvm_dev = RT_NULL;
#elif defined(RT_USING_SERIAL)
#include <rtdevice.h>
static struct rt_serial_device *gdb_serial;
#else
static rt_device_t gdb_dev = RT_NULL;
#endif


void gdb_uart_putc(char c);
int gdb_uart_getc();


/*if you want to use something instead of the serial,change it */
#ifdef RT_GDB_INTTER
struct gdb_io	gdb_io_ops;/* = {
    gdb_uart_getc,
    gdb_uart_putc
};*/
#else
struct gdb_io	gdb_io_ops = {
    gdb_uart_getc,
    gdb_uart_putc
};
#endif


/**
 * @ingroup gdb_stub
 *
 * This function will get GDB stubs started, with a proper environment
 */
void gdb_start()
{
#ifdef CONFIG_DEVICES_RVM_HAL
    if (gdb_rvm_dev == RT_NULL)
#elif defined(RT_GDB_INTTER)
    if(0)
#else
    if (gdb_dev == RT_NULL)
#endif
        rt_kprintf("GDB: no gdb_dev found,please set it first\n");
    else
        gdb_breakpoint();
}


/**
 * @ingroup gdb_stub
 *
 * This function sets the input device of gdb_stub.
 *
 * @param device_name the name of new input device.
 */
void gdb_set_device(const char* device_name)
{
#ifdef CONFIG_DEVICES_RVM_HAL
    gdb_rvm_dev = rvm_hal_uart_open(device_name);
    if (gdb_rvm_dev) {
        rvm_hal_uart_config_t config;
        config.baud_rate = 115200;
        config.mode = MODE_TX_RX;
        config.flow_control = FLOW_CONTROL_DISABLED;
        config.stop_bits = STOP_BITS_1;
        config.parity = PARITY_NONE;
        config.data_width = DATA_WIDTH_8BIT;
        rvm_hal_uart_config(gdb_rvm_dev, &config);
        rvm_hal_uart_set_type(gdb_rvm_dev, UART_TYPE_SYNC);
    }
#else
    rt_device_t dev = RT_NULL;
    dev = rt_device_find(device_name);
    if (dev == RT_NULL) {
        rt_kprintf("GDB: can not find device: %s\n", device_name);
        return;
    }

    /* open this device and set the new device  */
    if (rt_device_open(dev, RT_DEVICE_OFLAG_RDWR) == RT_EOK) {
        gdb_dev = dev;
#ifdef RT_USING_SERIAL
        gdb_serial = (struct rt_serial_device *)gdb_dev;
#endif
    }
#endif
}

void *gdb_get_device()
{
#ifdef CONFIG_DEVICES_RVM_HAL
    return gdb_rvm_dev;
#elif defined(RT_USING_SERIAL)
    return gdb_serial;
#elif defined(RT_GDB_INTTER)
    return (void*)0x1234;
#else
    return gdb_dev;
#endif
}

void gdb_uart_putc(char c)
{
#ifdef RT_GDB_DEBUG
    rt_kprintf("%c",c);
#endif
#ifdef CONFIG_DEVICES_RVM_HAL
    rvm_hal_uart_send(gdb_rvm_dev, &c, 1, AOS_WAIT_FOREVER);
#else
    rt_device_write(gdb_dev, 0, &c, 1);
#endif
}

/*  polling  */
int gdb_uart_getc()
{
    int ch;

#ifdef CONFIG_DEVICES_RVM_HAL
    rvm_hal_uart_recv_poll(gdb_rvm_dev, &ch, 1);
#elif defined(RT_USING_SERIAL)
    ch = -1;
    do {
        ch = gdb_serial->ops->getc(gdb_serial);
    } while (ch == -1);
#else
    int ret;
    do{
        ret = rt_device_read(gdb_dev, 0, &ch, 1);
    }while(ret != 1);
#endif

#ifdef RT_GDB_DEBUG
    rt_kprintf("%c",ch);
#endif

    return ch;
}

void gdb_set_iofunc(int(*read_char)(void), void(*write_char)(char))
{
    gdb_io_ops.read_char = read_char;
    gdb_io_ops.write_char = write_char;
}
