#ifndef _UTIL_H
#define _UTIL_H

#include <string>

#include <stdio.h>
#include <stdlib.h>

void printErrorAndExit(std::string err)
{
    printf("[Error] %s\n", err.c_str());
    exit(-1);
}

void printWarning(std::string warn)
{
    printf("[Warning] %s\n", warn.c_str());
}

#define CHECK(x)                              \
        if(!(x))                              \
            printErrorAndExit("check failed")
      
#endif // _UTIL_H