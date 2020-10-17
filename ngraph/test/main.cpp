//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <chrono>
#include <iostream>

#include "gtest/gtest.h"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "runtime/backend.hpp"
#include "runtime/backend_manager.hpp"
#include "util/test_environment.hpp"

using namespace std;

int main(int argc, char** argv)
{
    const string cpath_flag{"--cpath"};
    string cpath;
    const char* exclude = "--gtest_filter=-benchmark.*";
    vector<char*> argv_vector;
    argv_vector.push_back(argv[0]);
    argv_vector.push_back(const_cast<char*>(exclude));
    for (int i = 1; i < argc; i++)
    {
        argv_vector.push_back(argv[i]);
    }
    argc = argv_vector.size();
    ::testing::InitGoogleTest(&argc, argv_vector.data());
    ::testing::AddGlobalTestEnvironment(new ngraph::test::TestEnvironment);
    for (int i = 1; i < argc; i++)
    {
        if (cpath_flag == argv[i] && (++i) < argc)
        {
            cpath = argv[i];
        }
    }
    ngraph::runtime::Backend::set_backend_shared_library_search_directory(cpath);

    int rc = RUN_ALL_TESTS();

    return rc;
}
