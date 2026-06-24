#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <openssl/evp.h>
#include <openssl/md5.h>

#define MAJOR_VERSION 1
#define MINOR_VERSION 1
#define REVISON 0

#pragma pack(push, 1)
typedef struct custom_header {
    unsigned char magic[4];        // "TSM\x01"
    uint8_t major_version;
    uint8_t minor_version;
    uint16_t revision;
    uint8_t md5_sum[16];  // 原始文件的 MD5（字符串形式)
    uint8_t reserved_pad[16];
    unsigned char company_name[32];
    uint64_t timestamp;
    uint32_t header_size;
    uint32_t bin_file_size;
    uint16_t chip_type;  // c908 0x00
    uint16_t core_type;  // Kcore 0x01. Score0 0x02, Score1 0x03
    uint8_t reserved[35];
    uint8_t checksum;
} custom_header_t;
#pragma pack(pop)

// 计算文件MD5
void calc_md5(const char *filename, unsigned char *md5_str) {
    unsigned char md5[MD5_DIGEST_LENGTH];
    unsigned char buf[1024];
    size_t n;
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();

    EVP_DigestInit_ex(ctx, EVP_md5(), NULL);

    FILE *f = fopen(filename, "rb");
    if (!f) {
        perror("open");
        return;
    }
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) EVP_DigestUpdate(ctx, buf, n);
    fclose(f);

    EVP_DigestFinal_ex(ctx, md5, NULL);
    EVP_MD_CTX_free(ctx);
    
    for (int i = 0; i < MD5_DIGEST_LENGTH; ++i) {
        md5_str[i] = md5[i];
    }
    // for (int i = 0; i < MD5_DIGEST_LENGTH; i++) sprintf((char *)&md5_str[i * 2], "%02x", md5[i]);
}

// 计算头部校验和（除自身外所有字段的字节累加）
uint8_t calc_checksum(const custom_header_t *hdr)
{
    const unsigned char *p = (const unsigned char *)hdr;
    uint8_t sum = 0;
    for (size_t i = 0; i < sizeof(custom_header_t) - 1; i++)
        sum += p[i];
    return 256 - sum;
}

int main(int argc, char *argv[])
{
    if (argc < 3) {
        printf("Usage: %s <input.bin> <output.tufw>\n", argv[0]);
        return 1;
    }

    const char *infile = argv[1];
    const char *outfile = argv[2];

    FILE *f_in = fopen(infile, "rb");
    if (!f_in) { perror("open input"); return 1; }

    fseek(f_in, 0, SEEK_END);
    uint32_t bin_size = ftell(f_in);
    fseek(f_in, 0, SEEK_SET);

    unsigned char *bin_data = malloc(bin_size);
    fread(bin_data, 1, bin_size, f_in);
    fclose(f_in);

    // 构造header
    custom_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, "TSM\x01", 4);
    hdr.major_version = MAJOR_VERSION;
    hdr.minor_version = MINOR_VERSION;
    hdr.revision = REVISON;
    hdr.timestamp = (uint64_t)time(NULL);
    hdr.header_size = sizeof(custom_header_t);
    hdr.bin_file_size = bin_size;
    hdr.chip_type = 0x00;  // c908 0x00
    hdr.core_type = 0x01;  // Kcore 0x01. Score0 0x02, Score1 0x03
    strcpy((char *)hdr.company_name, "tsingmicro");
    printf("hdr.company_name: %s\n", hdr.company_name);
    calc_md5(infile, hdr.md5_sum);
    hdr.checksum = calc_checksum(&hdr);

    // 写出文件
    FILE *f_out = fopen(outfile, "wb");
    fwrite(&hdr, 1, sizeof(hdr), f_out);
    fwrite(bin_data, 1, bin_size, f_out);
    fclose(f_out);

    printf("合成完成：%s (含%d字节头)\n", outfile, hdr.header_size);
    free(bin_data);
    return 0;
}
