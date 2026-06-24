#ifndef _KERNEL_VERSION_H_
#define _KERNEL_VERSION_H_

/* The template values come from cmake/version.cmake
 * BUILD_VERSION related template values will be 'git describe',
 * alternatively user defined BUILD_VERSION.
 */

/* #undef ZEPHYR_VERSION_CODE */
/* #undef ZEPHYR_VERSION */

#define KERNELVERSION                   0x1000000
#define KERNEL_VERSION_NUMBER           0x10000
#define KERNEL_VERSION_MAJOR            1
#define KERNEL_VERSION_MINOR            0
#define KERNEL_PATCHLEVEL               0
#define KERNEL_TWEAK                    0
#define KERNEL_VERSION_STRING           "1.0.0"
#define KERNEL_VERSION_EXTENDED_STRING  ""
#define KERNEL_VERSION_TWEAK_STRING     ""

#define BUILD_VERSION Xuantie-gcc-newlib-v2.8.0-281-gb72af380588c


#endif /* _KERNEL_VERSION_H_ */
