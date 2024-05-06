// Main.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#define _DEPTH_DISABLED

#include <iostream>
#include <chrono>
#include <functional>
#include <unordered_map>
#include <csignal>
#include "StringTemplate.h"
#include "CGPStream.h"
#include "Cgp.h"
#include "Learning.h"

void signal_handler(int signum);
