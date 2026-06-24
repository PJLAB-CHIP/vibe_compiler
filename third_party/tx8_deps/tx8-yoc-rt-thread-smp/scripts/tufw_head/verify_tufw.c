// verify_tufw.c  — 校验 *.tufw 文件头与负载
#include <inttypes.h>
#include <openssl/evp.h>
#include <openssl/md5.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma pack(push, 1)
typedef struct custom_header {
    unsigned char magic[4];  // "TSM\x01"
    uint8_t major_version;
    uint8_t minor_version;
    uint16_t revision;
    uint8_t md5_sum[16];  // 原始bin的MD5(32字节ASCII十六进制)
    uint8_t reserved_pad[16];
    unsigned char company_name[32];
    uint64_t timestamp;
    uint32_t header_size;    // 固定应为 128
    uint32_t bin_file_size;  // 负载大小
    uint16_t chip_type;
    uint16_t core_type;
    uint8_t reserved[35];
    uint8_t checksum;  // 使“全头部字节和 %256 == 0”
} custom_header_t;
#pragma pack(pop)

// 计算整块内存的“字节和 % 256”
static uint8_t sum_mod256(const void *buf, size_t n) {
    const uint8_t *p = (const uint8_t *)buf;
    uint32_t s = 0;
    for (size_t i = 0; i < n; ++i) s += p[i];
    return (uint8_t)(s & 0xFF);  // 等价 % 256
}

// 计算文件某段的 MD5 并返回 32字节hex字符串（不带\0）
static int md5_region_hex(FILE *fp, long offset, size_t len, unsigned char out32[16]) {
    unsigned char md[MD5_DIGEST_LENGTH];
    unsigned char buf[1 << 15];
    size_t todo = len;

    if (fseek(fp, offset, SEEK_SET) != 0) return -1;

    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    if (!ctx) return -1;
    EVP_DigestInit_ex(ctx, EVP_md5(), NULL);

    while (todo > 0) {
        size_t chunk = todo > sizeof(buf) ? sizeof(buf) : todo;
        size_t n = fread(buf, 1, chunk, fp);
        if (n == 0 && ferror(fp)) {
            EVP_MD_CTX_free(ctx);
            return -1;
        }
        if (n == 0) break;
        EVP_DigestUpdate(ctx, buf, n);
        todo -= n;
    }
    EVP_DigestFinal_ex(ctx, md, NULL);
    EVP_MD_CTX_free(ctx);

    for (int i = 0; i < MD5_DIGEST_LENGTH; ++i) {
        out32[i] = md[i];
    }
    return 0;
}
static void print_header_info(const custom_header_t *hdr) {
    printf("\n===== TUFw Header Info =====\n");
    printf("Magic:            %c%c%c 0x%02X\n", hdr->magic[0], hdr->magic[1], hdr->magic[2], hdr->magic[3]);
    printf("Version:          %u.%u.%u\n", hdr->major_version, hdr->minor_version, hdr->revision);

    printf("MD5 (bin):        ");
    char md5_str[32];
    static const unsigned char hex[] = "0123456789abcdef";
    for (int i = 0; i < MD5_DIGEST_LENGTH; ++i) {
        md5_str[i * 2] = hex[hdr->md5_sum[i] >> 4];
        md5_str[i * 2 + 1] = hex[hdr->md5_sum[i] & 0xF];
    }
    for (int i = 0; i < 32; ++i) putchar(md5_str[i]);
    printf("\n");

    printf("Company:          %.*s\n", 32, hdr->company_name);
    printf("Timestamp:        %" PRIu64 "\n", hdr->timestamp);
    printf("Header size:      %u bytes\n", hdr->header_size);
    printf("Binary size:      %u bytes\n", hdr->bin_file_size);
    printf("Chip type:        0x%04X\n", hdr->chip_type);
    printf("Core type:        0x%04X\n", hdr->core_type);

    printf("Reserved:         ");
    for (int i = 0; i < sizeof(hdr->reserved); ++i) printf("%02X ", hdr->reserved[i]);
    printf("\n");

    printf("Checksum:         0x%02X\n", hdr->checksum);
    printf("============================\n\n");
}

static int verify_tufw(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        perror("open tufw");
        return -1;
    }

    custom_header_t hdr;
    if (fread(&hdr, 1, sizeof(hdr), fp) != sizeof(hdr)) {
        fprintf(stderr, "read header failed\n");
        fclose(fp);
        return -1;
    }
    printf("hdr.company_name: %s\n",(const char*) hdr.company_name);
    printf("Company (hex):    ");
    for (int i = 0; i < 32; ++i) printf("%02X ", hdr.company_name[i]);
    printf("\n");

    // 1) 魔数
    static const unsigned char expect_magic[4] = {'T', 'S', 'M', 0x01};
    if (memcmp(hdr.magic, expect_magic, 4) != 0) {
        fprintf(stderr, "magic mismatch\n");
        fclose(fp);
        return -1;
    }

    // 2) 头长
    if (hdr.header_size != sizeof(custom_header_t)) {
        fprintf(stderr, "header_size mismatch: %u (expect %zu)\n", hdr.header_size, sizeof(custom_header_t));
        fclose(fp);
        return -1;
    }

    // 3) checksum：包含 checksum 字段在内总和 %256 == 0
    uint8_t mod = sum_mod256(&hdr, sizeof(hdr));
    if (mod != 0) {
        fprintf(stderr, "header checksum invalid: sum%%256 = %u (expect 0)\n", mod);
        fclose(fp);
        return -1;
    }

    // 4) 文件长度与负载大小
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return -1;
    }
    long file_size = ftell(fp);
    if (file_size < 0) {
        fclose(fp);
        return -1;
    }
    long payload_off = (long)sizeof(custom_header_t);
    long payload_size = file_size - payload_off;
    if ((uint32_t)payload_size != hdr.bin_file_size) {
        fprintf(stderr, "bin size mismatch: header=%u, actual=%ld\n", hdr.bin_file_size, payload_size);
        fclose(fp);
        return -1;
    }

    // 5) MD5 校验
    unsigned char md5_hex[32];
    if (md5_region_hex(fp, payload_off, (size_t)payload_size, md5_hex) != 0) {
        fprintf(stderr, "md5 calc failed\n");
        fclose(fp);
        return -1;
    }
    for (int i = 0; i < MD5_DIGEST_LENGTH; ++i) {
        printf("%c", md5_hex[i]);
    }
    printf("\n");

    // 通过
    print_header_info(&hdr);
    printf("Verification passed !\n");
    fclose(fp);
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <file.tufw>\n", argv[0]);
        return 2;
    }
    return verify_tufw(argv[1]) == 0 ? 0 : 1;
}
