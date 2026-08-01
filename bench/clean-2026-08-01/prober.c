/* prober.c — what can this drive actually do with colibri's access pattern?
 *
 * Colibri streams routed experts as ~21.2 MB reads scattered across 144
 * safetensors shards, O_DIRECT, from N loader threads. This reproduces exactly
 * that and reports aggregate throughput, so "the drive is the limit" can be
 * tested against the pattern that matters rather than a sequential number.
 *
 * cc -O2 -pthread prober.c -o prober
 * ./prober <nthreads> <read_MB> <seconds> <dir>
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <dirent.h>

#define MAXF 256
static int   fds[MAXF];
static off_t sizes[MAXF];
static int   nf = 0;
static size_t RSZ;
static volatile int stop_now = 0;
static atomic_llong total_bytes;

static double now(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

static void *worker(void *arg) {
    unsigned seed = (unsigned)(size_t)arg * 2654435761u + 12345u;
    void *buf;
    if (posix_memalign(&buf, 4096, RSZ)) return NULL;   /* O_DIRECT needs alignment */
    while (!stop_now) {
        int i = rand_r(&seed) % nf;
        if (sizes[i] <= (off_t)RSZ) continue;
        off_t span = (sizes[i] - (off_t)RSZ) / 4096;
        off_t off  = ((off_t)(rand_r(&seed) % (span > 0 ? span : 1))) * 4096;
        ssize_t n = pread(fds[i], buf, RSZ, off);
        if (n > 0) atomic_fetch_add(&total_bytes, (long long)n);
        else if (n < 0) { perror("pread"); break; }
    }
    free(buf);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s nthreads read_MB seconds dir\n", argv[0]); return 1; }
    int nth = atoi(argv[1]);
    RSZ = (size_t)(atof(argv[2]) * 1048576.0);
    RSZ = (RSZ / 4096) * 4096;                          /* align the size too */
    double secs = atof(argv[3]);
    const char *dir = argv[4];

    DIR *d = opendir(dir);
    if (!d) { perror("opendir"); return 1; }
    struct dirent *e;
    char path[4096];
    while ((e = readdir(d)) && nf < MAXF) {
        if (!strstr(e->d_name, ".safetensors")) continue;
        snprintf(path, sizeof path, "%s/%s", dir, e->d_name);
        int fd = open(path, O_RDONLY | O_DIRECT);
        if (fd < 0) continue;
        struct stat st;
        if (fstat(fd, &st) == 0 && st.st_size > (off_t)RSZ) { fds[nf] = fd; sizes[nf] = st.st_size; nf++; }
        else close(fd);
    }
    closedir(d);
    if (!nf) { fprintf(stderr, "no readable shards in %s\n", dir); return 1; }

    atomic_store(&total_bytes, 0);
    pthread_t th[512];
    double t0 = now();
    for (int i = 0; i < nth; i++) pthread_create(&th[i], NULL, worker, (void *)(size_t)(i + 1));
    while (now() - t0 < secs) usleep(50000);
    stop_now = 1;
    for (int i = 0; i < nth; i++) pthread_join(th[i], NULL);
    double dt = now() - t0;

    double mb = (double)atomic_load(&total_bytes) / 1048576.0;
    printf("QD=%-3d read=%.1fMB  %7.0f MB/s  (%.1f GB in %.1fs, %d shards)\n",
           nth, RSZ / 1048576.0, mb / dt, mb / 1024.0, dt, nf);
    for (int i = 0; i < nf; i++) close(fds[i]);
    return 0;
}
