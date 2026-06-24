/*
 * Copyright (c) 2006-2021, RT-Thread Development Team
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author          Notes
 * 2018/08/29     Bernard         first version
 * 2021/04/23     chunyexixiaoyu  distinguish 32-bit and 64-bit
 */

#ifndef DL_ELF_H__
#define DL_ELF_H__
#include <rtdef.h>
typedef rt_uint8_t              Elf_Byte;

typedef rt_uint32_t             Elf32_Addr;    /* Unsigned program address */
typedef rt_uint32_t             Elf32_Off;     /* Unsigned file offset */
typedef rt_int32_t              Elf32_Sword;   /* Signed large integer */
typedef rt_uint32_t             Elf32_Word;    /* Unsigned large integer */
typedef rt_uint16_t             Elf32_Half;    /* Unsigned medium integer */

typedef rt_uint64_t             Elf64_Addr;
typedef rt_uint16_t             Elf64_Half;
typedef rt_int16_t              Elf64_SHalf;
typedef rt_uint64_t             Elf64_Off;
typedef rt_int32_t              Elf64_Sword;
typedef rt_uint32_t             Elf64_Word;
typedef rt_uint64_t             Elf64_Xword;
typedef rt_int64_t              Elf64_Sxword;
typedef rt_uint16_t             Elf64_Section;

/* e_ident[] magic number */
#define ELFMAG0                 0x7f           /* e_ident[EI_MAG0] */
#define ELFMAG1                 'E'            /* e_ident[EI_MAG1] */
#define ELFMAG2                 'L'            /* e_ident[EI_MAG2] */
#define ELFMAG3                 'F'            /* e_ident[EI_MAG3] */
#define RTMMAG                  "\177RTM"      /* magic */
#define ELFMAG                  "\177ELF"      /* magic */
#define SELFMAG                 4              /* size of magic */

#define EI_CLASS                4              /* file class */
#define EI_NIDENT               16             /* Size of e_ident[] */

/* e_ident[] file class */
#define ELFCLASSNONE            0              /* invalid */
#define ELFCLASS32              1              /* 32-bit objs */
#define ELFCLASS64              2              /* 64-bit objs */
#define ELFCLASSNUM             3              /* number of classes */

/* e_ident[] data encoding */
#define ELFDATANONE             0              /* invalid */
#define ELFDATA2LSB             1              /* Little-Endian */
#define ELFDATA2MSB             2              /* Big-Endian */
#define ELFDATANUM              3              /* number of data encode defines */

/* e_ident */
#define IS_ELF(ehdr)            ((ehdr).e_ident[EI_MAG0] == ELFMAG0 && \
                                 (ehdr).e_ident[EI_MAG1] == ELFMAG1 && \
                                 (ehdr).e_ident[EI_MAG2] == ELFMAG2 && \
                                 (ehdr).e_ident[EI_MAG3] == ELFMAG3)

#define ET_NONE                 0              /* No file type */
#define ET_REL                  1              /* Relocatable file */
#define ET_EXEC                 2              /* Executable file */
#define ET_DYN                  3              /* Shared object file */
#define ET_CORE                 4              /* Core file */

/* ELF Header */
typedef struct elfhdr
{
    unsigned char e_ident[EI_NIDENT];          /* ELF Identification */
    Elf32_Half    e_type;                      /* object file type */
    Elf32_Half    e_machine;                   /* machine */
    Elf32_Word    e_version;                   /* object file version */
    Elf32_Addr    e_entry;                     /* virtual entry point */
    Elf32_Off     e_phoff;                     /* program header table offset */
    Elf32_Off     e_shoff;                     /* section header table offset */
    Elf32_Word    e_flags;                     /* processor-specific flags */
    Elf32_Half    e_ehsize;                    /* ELF header size */
    Elf32_Half    e_phentsize;                 /* program header entry size */
    Elf32_Half    e_phnum;                     /* number of program header entries */
    Elf32_Half    e_shentsize;                 /* section header entry size */
    Elf32_Half    e_shnum;                     /* number of section header entries */
    Elf32_Half    e_shstrndx;                  /* section header table's "section
                                                  header string table" entry offset */
} Elf32_Ehdr;

typedef struct elf64_hdr
{
    unsigned char     e_ident[EI_NIDENT];        /* ELF Identification */
    Elf64_Half        e_type;                    /* object file type */
    Elf64_Half        e_machine;                 /* machine */
    Elf64_Word        e_version;                 /* object file version */
    Elf64_Addr        e_entry;                   /* virtual entry point */
    Elf64_Off         e_phoff;                   /* program header table offset */
    Elf64_Off         e_shoff;                   /* section header table offset */
    Elf64_Word        e_flags;                   /* processor-specific flags */
    Elf64_Half        e_ehsize;                  /* ELF header size */
    Elf64_Half        e_phentsize;               /* program header entry size */
    Elf64_Half        e_phnum;                   /* number of program header entries */
    Elf64_Half        e_shentsize;               /* section header entry size */
    Elf64_Half        e_shnum;                   /* number of section header entries */
    Elf64_Half        e_shstrndx;                /* section header table's "section
                                                header string table" entry offset */
} Elf64_Ehdr;

/* Section Header */
typedef struct
{
    Elf32_Word sh_name;                        /* name - index into section header
                                                  string table section */
    Elf32_Word sh_type;                        /* type */
    Elf32_Word sh_flags;                       /* flags */
    Elf32_Addr sh_addr;                        /* address */
    Elf32_Off  sh_offset;                      /* file offset */
    Elf32_Word sh_size;                        /* section size */
    Elf32_Word sh_link;                        /* section header table index link */
    Elf32_Word sh_info;                        /* extra information */
    Elf32_Word sh_addralign;                   /* address alignment */
    Elf32_Word sh_entsize;                     /* section entry size */
} Elf32_Shdr;

typedef struct
{
    Elf64_Word        sh_name;                   /* Section name (string tbl index) */
    Elf64_Word        sh_type;                   /* Section type */
    Elf64_Xword       sh_flags;                  /* Section flags */
    Elf64_Addr        sh_addr;                   /* Section virtual addr at execution */
    Elf64_Off         sh_offset;                 /* Section file offset */
    Elf64_Xword       sh_size;                   /* Section size in bytes */
    Elf64_Word        sh_link;                   /* Link to another section */
    Elf64_Word        sh_info;                   /* Additional section information */
    Elf64_Xword       sh_addralign;              /* Section alignment */
    Elf64_Xword       sh_entsize;                /* Entry size if section holds table */
} Elf64_Shdr;

/* Section names */
#define ELF_BSS                 ".bss"         /* uninitialized data */
#define ELF_DATA                ".data"        /* initialized data */
#define ELF_DEBUG               ".debug"       /* debug */
#define ELF_DYNAMIC             ".dynamic"     /* dynamic linking information */
#define ELF_DYNSTR              ".dynstr"      /* dynamic string table */
#define ELF_DYNSYM              ".dynsym"      /* dynamic symbol table */
#define ELF_FINI                ".fini"        /* termination code */
#define ELF_GOT                 ".got"         /* global offset table */
#define ELF_HASH                ".hash"        /* symbol hash table */
#define ELF_INIT                ".init"        /* initialization code */
#define ELF_REL_DATA            ".rel.data"    /* relocation data */
#define ELF_REL_FINI            ".rel.fini"    /* relocation termination code */
#define ELF_REL_INIT            ".rel.init"    /* relocation initialization code */
#define ELF_REL_DYN             ".rel.dyn"     /* relocaltion dynamic link info */
#define ELF_REL_RODATA          ".rel.rodata"  /* relocation read-only data */
#define ELF_REL_TEXT            ".rel.text"    /* relocation code */
#define ELF_RODATA              ".rodata"      /* read-only data */
#define ELF_SHSTRTAB            ".shstrtab"    /* section header string table */
#define ELF_STRTAB              ".strtab"      /* string table */
#define ELF_SYMTAB              ".symtab"      /* symbol table */
#define ELF_TEXT                ".text"        /* code */
#define ELF_RTMSYMTAB           "RTMSymTab"
#define ELF_EXPORTED_SYMTAB     "ExportedDYNSYMTab"      /* dyna4mic symbol table */
#define ELF_CONTROL_SYMTAB      "ControlSYMTab"      /* */

/* Symbol Table Entry */
typedef struct elf32_sym
{
    Elf32_Word    st_name;                     /* name - index into string table */
    Elf32_Addr    st_value;                    /* symbol value */
    Elf32_Word    st_size;                     /* symbol size */
    unsigned char st_info;                     /* type and binding */
    unsigned char st_other;                    /* 0 - no defined meaning */
    Elf32_Half    st_shndx;                    /* section header index */
} Elf32_Sym;

typedef struct
{
    Elf64_Word        st_name;                   /* Symbol name (string tbl index) */
    unsigned char     st_info;                   /* Symbol type and binding */
    unsigned char     st_other;                  /* Symbol visibility */
    Elf64_Section     st_shndx;                  /* Section index */
    Elf64_Addr        st_value;                  /* Symbol value */
    Elf64_Xword       st_size;                   /* Symbol size */
} Elf64_Sym;

#define STB_LOCAL               0              /* BIND */
#define STB_GLOBAL              1
#define STB_WEAK                2
#define STB_NUM                 3

#define STB_LOPROC              13             /* processor specific range */
#define STB_HIPROC              15

#define STT_NOTYPE              0              /* symbol type is unspecified */
#define STT_OBJECT              1              /* data object */
#define STT_FUNC                2              /* code object */
#define STT_SECTION             3              /* symbol identifies an ELF section */
#define STT_FILE                4              /* symbol's name is file name */
#define STT_COMMON              5              /* common data object */
#define STT_TLS                 6              /* thread-local data object */
#define STT_NUM                 7              /* # defined types in generic range */
#define STT_LOOS                10             /* OS specific range */
#define STT_HIOS                12
#define STT_LOPROC              13             /* processor specific range */
#define STT_HIPROC              15

#define STN_UNDEF               0              /* undefined */

#define ELF_ST_BIND(info)       ((info) >> 4)
#define ELF_ST_TYPE(info)       ((info) & 0xf)
#define ELF_ST_INFO(bind, type) (((bind)<<4)+((type)&0xf))

/* Relocation entry with implicit addend */
typedef struct
{
    Elf32_Addr r_offset;                       /* offset of relocation */
    Elf32_Word r_info;                         /* symbol table index and type */
} Elf32_Rel;

typedef struct
{
    Elf64_Addr   r_offset;                     /* Address */
    Elf64_Xword  r_info;                       /* Relocation type and symbol index */
} Elf64_Rel;

/* Relocation entry with explicit addend */
typedef struct
{
    Elf32_Addr  r_offset;                      /* offset of relocation */
    Elf32_Word  r_info;                        /* symbol table index and type */
    Elf32_Sword r_addend;
} Elf32_Rela;

typedef struct
{
    Elf64_Addr      r_offset;                  /* Address */
    Elf64_Xword     r_info;                    /* Relocation type and symbol index */
    Elf64_Sxword    r_addend;                  /* Addend */
} Elf64_Rela;

/* Extract relocation info - r_info */
#define ELF32_R_SYM(i)          ((i) >> 8)
#define ELF32_R_TYPE(i)         ((unsigned char) (i))
#define ELF32_R_INFO(s,t)       (((s) << 8) + (unsigned char)(t))

#define ELF64_R_SYM(i)                        ((i) >> 32)
#define ELF64_R_TYPE(i)                        ((i) & 0xffffffff)
#define ELF64_R_INFO(sym,type)                ((((Elf64_Xword) (sym)) << 32) + (type))

/*
 * Relocation type for arm
 */
#define R_ARM_NONE              0
#define R_ARM_PC24              1
#define R_ARM_ABS32             2
#define R_ARM_REL32             3
#define R_ARM_THM_CALL          10
#define R_ARM_GLOB_DAT          21
#define R_ARM_JUMP_SLOT         22
#define R_ARM_RELATIVE          23
#define R_ARM_GOT_BREL          26
#define R_ARM_PLT32             27
#define R_ARM_CALL              28
#define R_ARM_JUMP24            29
#define R_ARM_THM_JUMP24        30
#define R_ARM_V4BX              40

/*
 * Relocation type for x86
 */
#define R_386_NONE              0
#define R_386_32                1
#define R_386_PC32              2
#define R_386_GOT32             3
#define R_386_PLT32             4
#define R_386_COPY              5
#define R_386_GLOB_DAT          6
#define R_386_JMP_SLOT          7
#define R_386_RELATIVE          8
#define R_386_GOTOFF            9
#define R_386_GOTPC             10

/* Program Header */
typedef struct
{
    Elf32_Word p_type;                         /* segment type */
    Elf32_Off  p_offset;                       /* segment offset */
    Elf32_Addr p_vaddr;                        /* virtual address of segment */
    Elf32_Addr p_paddr;                        /* physical address - ignored? */
    Elf32_Word p_filesz;                       /* number of bytes in file for seg. */
    Elf32_Word p_memsz;                        /* number of bytes in mem. for seg. */
    Elf32_Word p_flags;                        /* flags */
    Elf32_Word p_align;                        /* memory alignment */
} Elf32_Phdr;

typedef struct
{
    Elf64_Word   p_type;                       /* Segment type */
    Elf64_Word   p_flags;                      /* Segment flags */
    Elf64_Off    p_offset;                     /* Segment file offset */
    Elf64_Addr   p_vaddr;                      /* Segment virtual address */
    Elf64_Addr   p_paddr;                      /* Segment physical address */
    Elf64_Xword  p_filesz;                     /* Segment size in file */
    Elf64_Xword  p_memsz;                      /* Segment size in memory */
    Elf64_Xword  p_align;                      /* Segment alignment */
} Elf64_Phdr;

/* p_type */
#define PT_NULL                 0
#define PT_LOAD                 1
#define PT_DYNAMIC              2
#define PT_INTERP               3
#define PT_NOTE                 4
#define PT_SHLIB                5
#define PT_PHDR                 6
#define PT_TLS                  7
#define PT_NUM                  8
#define PT_LOOS                 0x60000000
#define PT_HIOS                 0x6fffffff
#define PT_LOPROC               0x70000000
#define PT_HIPROC               0x7fffffff

/* p_flags */
#define PF_X                    1
#define PF_W                    2
#define PF_R                    4

/* sh_type */
#define SHT_NULL                0              /* inactive */
#define SHT_PROGBITS            1              /* program defined information */
#define SHT_SYMTAB              2              /* symbol table section */
#define SHT_STRTAB              3              /* string table section */
#define SHT_RELA                4              /* relocation section with addends*/
#define SHT_HASH                5              /* symbol hash table section */
#define SHT_DYNAMIC             6              /* dynamic section */
#define SHT_NOTE                7              /* note section */
#define SHT_NOBITS              8              /* no space section */
#define SHT_REL                 9              /* relocation section without addends */
#define SHT_SHLIB               10             /* reserved - purpose unknown */
#define SHT_DYNSYM              11             /* dynamic symbol table section */
#define SHT_NUM                 12             /* number of section types */
#define SHT_LOPROC              0x70000000     /* reserved range for processor */
#define SHT_HIPROC              0x7fffffff     /* specific section header types */
#define SHT_LOUSER              0x80000000     /* reserved range for application */
#define SHT_HIUSER              0xffffffff     /* specific indexes */

/* Section Attribute Flags - sh_flags */
#define SHF_WRITE               0x1            /* Writable */
#define SHF_ALLOC               0x2            /* occupies memory */
#define SHF_EXECINSTR           0x4            /* executable */
#define SHF_MASKPROC            0xf0000000     /* reserved bits for processor */
/* specific section attributes */


/* Dynamic section entry.  */

typedef struct
{
    Elf32_Sword	d_tag;			/* Dynamic entry type */
    union
    {
        Elf32_Word d_val;			/* Integer value */
        Elf32_Addr d_ptr;			/* Address value */
    } d_un;
} Elf32_Dyn;

typedef struct
{
    Elf64_Sxword	d_tag;			/* Dynamic entry type */
    union
    {
        Elf64_Xword d_val;		/* Integer value */
        Elf64_Addr d_ptr;			/* Address value */
    } d_un;
} Elf64_Dyn;


/* Legal values for d_tag (dynamic entry type).  */

#define DT_NULL		0		/* Marks end of dynamic section */
#define DT_NEEDED	1		/* Name of needed library */
#define DT_PLTRELSZ	2		/* Size in bytes of PLT relocs */
#define DT_PLTGOT	3		/* Processor defined value */
#define DT_HASH		4		/* Address of symbol hash table */
#define DT_STRTAB	5		/* Address of string table */
#define DT_SYMTAB	6		/* Address of symbol table */
#define DT_RELA		7		/* Address of Rela relocs */
#define DT_RELASZ	8		/* Total size of Rela relocs */
#define DT_RELAENT	9		/* Size of one Rela reloc */
#define DT_STRSZ	10		/* Size of string table */
#define DT_SYMENT	11		/* Size of one symbol table entry */
#define DT_INIT		12		/* Address of init function */
#define DT_FINI		13		/* Address of termination function */
#define DT_SONAME	14		/* Name of shared object */
#define DT_RPATH	15		/* Library search path (deprecated) */
#define DT_SYMBOLIC	16		/* Start symbol search here */
#define DT_REL		17		/* Address of Rel relocs */
#define DT_RELSZ	18		/* Total size of Rel relocs */
#define DT_RELENT	19		/* Size of one Rel reloc */
#define DT_PLTREL	20		/* Type of reloc in PLT */
#define DT_DEBUG	21		/* For debugging; unspecified */
#define DT_TEXTREL	22		/* Reloc might modify .text */
#define DT_JMPREL	23		/* Address of PLT relocs */
#define	DT_BIND_NOW	24		/* Process relocations of object */
#define	DT_INIT_ARRAY	25		/* Array with addresses of init fct */
#define	DT_FINI_ARRAY	26		/* Array with addresses of fini fct */
#define	DT_INIT_ARRAYSZ	27		/* Size in bytes of DT_INIT_ARRAY */
#define	DT_FINI_ARRAYSZ	28		/* Size in bytes of DT_FINI_ARRAY */
#define DT_RUNPATH	29		/* Library search path */
#define DT_FLAGS	30		/* Flags for the object being loaded */
#define DT_ENCODING	32		/* Start of encoded range */
#define DT_PREINIT_ARRAY 32		/* Array with addresses of preinit fct*/
#define DT_PREINIT_ARRAYSZ 33		/* size in bytes of DT_PREINIT_ARRAY */
#define DT_SYMTAB_SHNDX	34		/* Address of SYMTAB_SHNDX section */
#define	DT_NUM		35		/* Number used */
#define DT_LOOS		0x6000000d	/* Start of OS-specific */
#define DT_HIOS		0x6ffff000	/* End of OS-specific */
#define DT_LOPROC	0x70000000	/* Start of processor-specific */
#define DT_HIPROC	0x7fffffff	/* End of processor-specific */
#define	DT_PROCNUM	DT_MIPS_NUM	/* Most used by any processor */

/* DT_* entries which fall between DT_VALRNGHI & DT_VALRNGLO use the
   Dyn.d_un.d_val field of the Elf*_Dyn structure.  This follows Sun's
   approach.  */
#define DT_VALRNGLO	0x6ffffd00
#define DT_GNU_PRELINKED 0x6ffffdf5	/* Prelinking timestamp */
#define DT_GNU_CONFLICTSZ 0x6ffffdf6	/* Size of conflict section */
#define DT_GNU_LIBLISTSZ 0x6ffffdf7	/* Size of library list */
#define DT_CHECKSUM	0x6ffffdf8
#define DT_PLTPADSZ	0x6ffffdf9
#define DT_MOVEENT	0x6ffffdfa
#define DT_MOVESZ	0x6ffffdfb
#define DT_FEATURE_1	0x6ffffdfc	/* Feature selection (DTF_*).  */
#define DT_POSFLAG_1	0x6ffffdfd	/* Flags for DT_* entries, effecting
					   the following DT_* entry.  */
#define DT_SYMINSZ	0x6ffffdfe	/* Size of syminfo table (in bytes) */
#define DT_SYMINENT	0x6ffffdff	/* Entry size of syminfo */
#define DT_VALRNGHI	0x6ffffdff
#define DT_VALTAGIDX(tag)	(DT_VALRNGHI - (tag))	/* Reverse order! */
#define DT_VALNUM 12

/* DT_* entries which fall between DT_ADDRRNGHI & DT_ADDRRNGLO use the
   Dyn.d_un.d_ptr field of the Elf*_Dyn structure.

   If any adjustment is made to the ELF object after it has been
   built these entries will need to be adjusted.  */
#define DT_ADDRRNGLO	0x6ffffe00
#define DT_GNU_HASH	0x6ffffef5	/* GNU-style hash table.  */
#define DT_TLSDESC_PLT	0x6ffffef6
#define DT_TLSDESC_GOT	0x6ffffef7
#define DT_GNU_CONFLICT	0x6ffffef8	/* Start of conflict section */
#define DT_GNU_LIBLIST	0x6ffffef9	/* Library list */
#define DT_CONFIG	0x6ffffefa	/* Configuration information.  */
#define DT_DEPAUDIT	0x6ffffefb	/* Dependency auditing.  */
#define DT_AUDIT	0x6ffffefc	/* Object auditing.  */
#define	DT_PLTPAD	0x6ffffefd	/* PLT padding.  */
#define	DT_MOVETAB	0x6ffffefe	/* Move table.  */
#define DT_SYMINFO	0x6ffffeff	/* Syminfo table.  */
#define DT_ADDRRNGHI	0x6ffffeff
#define DT_ADDRTAGIDX(tag)	(DT_ADDRRNGHI - (tag))	/* Reverse order! */
#define DT_ADDRNUM 11

/* The versioning entry types.  The next are defined as part of the
   GNU extension.  */
#define DT_VERSYM	0x6ffffff0

#define DT_RELACOUNT	0x6ffffff9
#define DT_RELCOUNT	0x6ffffffa

/* These were chosen by Sun.  */
#define DT_FLAGS_1	0x6ffffffb	/* State flags, see DF_1_* below.  */
#define	DT_VERDEF	0x6ffffffc	/* Address of version definition
					   table */
#define	DT_VERDEFNUM	0x6ffffffd	/* Number of version definitions */
#define	DT_VERNEED	0x6ffffffe	/* Address of table with needed
					   versions */
#define	DT_VERNEEDNUM	0x6fffffff	/* Number of needed versions */
#define DT_VERSIONTAGIDX(tag)	(DT_VERNEEDNUM - (tag))	/* Reverse order! */
#define DT_VERSIONTAGNUM 16

/* Sun added these machine-independent extensions in the "processor-specific"
   range.  Be compatible.  */
#define DT_AUXILIARY    0x7ffffffd      /* Shared object to load before self */
#define DT_FILTER       0x7fffffff      /* Shared object to get values from */
#define DT_EXTRATAGIDX(tag)	((Elf32_Word)-((Elf32_Sword) (tag) <<1>>1)-1)
#define DT_EXTRANUM	3

/* Values of `d_un.d_val' in the DT_FLAGS entry.  */
#define DF_ORIGIN	0x00000001	/* Object may use DF_ORIGIN */
#define DF_SYMBOLIC	0x00000002	/* Symbol resolutions starts here */
#define DF_TEXTREL	0x00000004	/* Object contains text relocations */
#define DF_BIND_NOW	0x00000008	/* No lazy binding for this object */
#define DF_STATIC_TLS	0x00000010	/* Module uses the static TLS model */

/* State flags selectable in the `d_un.d_val' element of the DT_FLAGS_1
   entry in the dynamic section.  */
#define DF_1_NOW	0x00000001	/* Set RTLD_NOW for this object.  */
#define DF_1_GLOBAL	0x00000002	/* Set RTLD_GLOBAL for this object.  */
#define DF_1_GROUP	0x00000004	/* Set RTLD_GROUP for this object.  */
#define DF_1_NODELETE	0x00000008	/* Set RTLD_NODELETE for this object.*/
#define DF_1_LOADFLTR	0x00000010	/* Trigger filtee loading at runtime.*/
#define DF_1_INITFIRST	0x00000020	/* Set RTLD_INITFIRST for this object*/
#define DF_1_NOOPEN	0x00000040	/* Set RTLD_NOOPEN for this object.  */
#define DF_1_ORIGIN	0x00000080	/* $ORIGIN must be handled.  */
#define DF_1_DIRECT	0x00000100	/* Direct binding enabled.  */
#define DF_1_TRANS	0x00000200
#define DF_1_INTERPOSE	0x00000400	/* Object is used to interpose.  */
#define DF_1_NODEFLIB	0x00000800	/* Ignore default lib search path.  */
#define DF_1_NODUMP	0x00001000	/* Object can't be dldump'ed.  */
#define DF_1_CONFALT	0x00002000	/* Configuration alternative created.*/
#define DF_1_ENDFILTEE	0x00004000	/* Filtee terminates filters search. */
#define	DF_1_DISPRELDNE	0x00008000	/* Disp reloc applied at build time. */
#define	DF_1_DISPRELPND	0x00010000	/* Disp reloc applied at run-time.  */
#define	DF_1_NODIRECT	0x00020000	/* Object has no-direct binding. */
#define	DF_1_IGNMULDEF	0x00040000
#define	DF_1_NOKSYMS	0x00080000
#define	DF_1_NOHDR	0x00100000
#define	DF_1_EDITED	0x00200000	/* Object is modified after built.  */
#define	DF_1_NORELOC	0x00400000
#define	DF_1_SYMINTPOSE	0x00800000	/* Object has individual interposers.  */
#define	DF_1_GLOBAUDIT	0x01000000	/* Global auditing required.  */
#define	DF_1_SINGLETON	0x02000000	/* Singleton symbols are used.  */
#define	DF_1_STUB	0x04000000
#define	DF_1_PIE	0x08000000
#define	DF_1_KMOD       0x10000000
#define	DF_1_WEAKFILTER 0x20000000
#define	DF_1_NOCOMMON   0x40000000

/* Flags for the feature selection in DT_FEATURE_1.  */
#define DTF_1_PARINIT	0x00000001
#define DTF_1_CONFEXP	0x00000002

/* Flags in the DT_POSFLAG_1 entry effecting only the next DT_* entry.  */
#define DF_P1_LAZYLOAD	0x00000001	/* Lazyload following object.  */
#define DF_P1_GROUPPERM	0x00000002	/* Symbols from next object are not
					   generally available.  */


#define CTR_EXPORT_SYM_POLICY	0x00000001
#define CTR_SO_TYPE	            0x00000002
#define CTR_SO_FILE_PATH	    0x00000003
#define CTR_SO_DEPS_FILE_PATH	0x00000004

#define CTR_SO_TYPE_SYSTEM_COMPONENT	0x00000001
#define EXPORT_ALL_SYM_POLICY	        0x00000001

#define IS_PROG(s)        (s.sh_type == SHT_PROGBITS)
#define IS_NOPROG(s)      (s.sh_type == SHT_NOBITS)
#define IS_REL(s)         (s.sh_type == SHT_REL)
#define IS_RELA(s)        (s.sh_type == SHT_RELA)
#define IS_ALLOC(s)       (s.sh_flags == SHF_ALLOC)
#define IS_AX(s)          ((s.sh_flags & SHF_ALLOC) && (s.sh_flags & SHF_EXECINSTR))
#define IS_AW(s)          ((s.sh_flags & SHF_ALLOC) && (s.sh_flags & SHF_WRITE))

#if (defined(__arm__) || defined(__i386__) || (__riscv_xlen == 32))
#define elf_module        ((Elf32_Ehdr *)module_ptr)
#define shdr              ((Elf32_Shdr *)((rt_uint8_t *)module_ptr + elf_module->e_shoff))
#define phdr              ((Elf32_Phdr *)((rt_uint8_t *)module_ptr + elf_module->e_phoff))

typedef Elf32_Sym       Elf_Sym;
typedef Elf32_Rel       Elf_Rel;
typedef Elf32_Addr      Elf_Addr;
#elif (defined(__aarch64__) || defined(__x86_64__) || (__riscv_xlen == 64))
#define elf_module        ((Elf64_Ehdr *)module_ptr)
#define shdr              ((Elf64_Shdr *)((rt_uint8_t *)module_ptr + elf_module->e_shoff))
#define phdr              ((Elf64_Phdr *)((rt_uint8_t *)module_ptr + elf_module->e_phoff))

typedef Elf64_Sym       Elf_Sym;
typedef Elf64_Rela      Elf_Rel;
typedef Elf64_Addr      Elf_Addr;
#endif

struct rt_dlmodule;
int dlmodule_relocate(struct rt_dlmodule *module, Elf_Rel *rel, Elf_Addr sym_val);
rt_err_t dlmodule_load_shared_object(struct rt_dlmodule *module, void *module_ptr);
rt_err_t dlmodule_load_relocated_object(struct rt_dlmodule *module, void *module_ptr);

#endif
