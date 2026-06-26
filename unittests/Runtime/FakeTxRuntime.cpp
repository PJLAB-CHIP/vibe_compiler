extern "C" int txSetDevice(int) { return 0; }
extern "C" int txMalloc(void **, unsigned long) { return 0; }
extern "C" int txFree(void *) { return 0; }
extern "C" int txMemcpy(void *, const void *, unsigned long, int) { return 0; }
extern "C" int txStreamSynchronize(void *) { return 0; }
extern "C" int txModuleLoad(void **, const char *) { return 0; }
extern "C" int txModuleGetFunction(void **, void *, const char *) { return 0; }
extern "C" int txLaunchKernel(void *, unsigned int, unsigned int, unsigned int,
                              unsigned int, unsigned int, unsigned int,
                              unsigned int, void *, void **, void **) {
  return 0;
}
