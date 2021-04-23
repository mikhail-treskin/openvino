// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gflags/gflags.h>
#include <iostream>

static const char help_message[] = "Print a usage message.";
static const char input_folders_message[] = "Required. Comma separated paths to the input folders with IRs";
static const char output_folder_message[] = "Required. Path to the output folders with IRs";
static const char constants_size_threshold_message[] = "Optional. Maximum size of constant to be serialized.\n"
                                                       "If constant size exceeds specified number it will be replaced"
                                                       "with parameter and meta information about original data range "
                                                       "will be saved";

DEFINE_bool(h, false, help_message);
DEFINE_string(input_folders, ".", input_folders_message);
DEFINE_string(output_folder, "output", output_folder_message);
DEFINE_double(constants_size_threshold, 1., constants_size_threshold_message);

/**
* @brief This function shows a help message
*/
static void showUsage() {
    std::cout << "\n";
    std::cout << "Subgraph Dumper [OPTION]\n";
    std::cout << "Options:\n";
    std::cout << "\n";
    std::cout << "    -h                                     " << help_message << "\n";
    std::cout << "    --input_folders \"<path>\"             " << input_folders_message << "\n";
    std::cout << "    --output_folder \"<path>\"             " << output_folder_message << "\n";
    std::cout << "    --constants_size_threshold \"<value>\" " << constants_size_threshold_message << "\n";
    std::cout << std::flush;
}