#include <fstream>
#include <iostream>

#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <execinfo.h>

#include "compiler/inc/stargazer.hpp"



/*
 * Print current stack trace to stderr
 */

static inline void print_stack_trace()
{
    enum {
        BACKTRACE_COUNT = 20
    };

    void *pc[BACKTRACE_COUNT];
    int count;

    /* Walk the stack and get all PCs */
    count = backtrace(pc, sizeof(pc) / sizeof(pc[0]));
    /* Translate the PCs to symbol and print them in STDERR*/
    backtrace_symbols_fd(pc, count, STDERR_FILENO);
}

/* signal handler.
 * Currently just prints the stack trace.
 */
static void sig_handler(int /*sig*/, siginfo_t* /*si*/, void* /*unused*/)
{
    print_stack_trace();
    exit(1);
}


int
main(int argc, char* argv[])
{
    // Setup signal handler: stack trace printing
    struct sigaction sigact = {};
    sigact.sa_flags = SA_SIGINFO;
    sigact.sa_sigaction = sig_handler;
    if (sigaction(SIGSEGV, &sigact, NULL)) {
        std::cout << "sigaction(SIGSEGV) error: " << errno;
    }
    if (sigaction(SIGINT, &sigact, NULL)) {
        std::cout << "sigaction(SIGINT) error: " << errno;
    }
    if (sigaction(SIGTERM, &sigact, NULL)) {
        std::cout << "sigaction(SIGTERM) error: " << errno;
    }

    // compilation
    kcc::compiler::Stargazer stargazer;
    return stargazer.Main(argc, argv);
}


