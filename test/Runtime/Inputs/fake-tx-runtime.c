int txSetDevice(int device) {
  (void)device;
  return 0;
}
int txMalloc(void **ptr, unsigned long bytes) {
  (void)ptr;
  (void)bytes;
  return 0;
}
int txFree(void *ptr) {
  (void)ptr;
  return 0;
}
int txMemcpy(void *dst, const void *src, unsigned long bytes, int kind) {
  (void)dst;
  (void)src;
  (void)bytes;
  (void)kind;
  return 0;
}
int txStreamSynchronize(void *stream) {
  (void)stream;
  return 0;
}
int txModuleLoad(void **module, const char *path) {
  (void)module;
  (void)path;
  return 0;
}
int txModuleGetFunction(void **function, void *module, const char *name) {
  (void)function;
  (void)module;
  (void)name;
  return 0;
}
int txLaunchKernel(void *function, unsigned int gridX, unsigned int gridY,
                   unsigned int gridZ, unsigned int blockX, unsigned int blockY,
                   unsigned int blockZ, unsigned int sharedMemBytes,
                   void *stream, void **args, void **extra) {
  (void)function;
  (void)gridX;
  (void)gridY;
  (void)gridZ;
  (void)blockX;
  (void)blockY;
  (void)blockZ;
  (void)sharedMemBytes;
  (void)stream;
  (void)args;
  (void)extra;
  return 0;
}
