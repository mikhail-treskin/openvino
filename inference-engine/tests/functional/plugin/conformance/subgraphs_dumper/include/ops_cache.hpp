// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>
#include <vector>
#include <string>
#include <memory>
#include <ngraph/ngraph.hpp>
#include "matchers/matchers_manager.hpp"

namespace SubgraphsDumper {
struct OPMetaInfo {
    std::string source_model;
    std::map<std::string, size_t> found_in_models;
    std::map<size_t, std::pair<double, double>> ports_ranges;
    std::vector<bool> param_to_constant_mask;

    explicit OPMetaInfo(const std::string &source_model) : source_model(source_model) {
        found_in_models = {{source_model, 1}};
        ports_ranges = {};
        param_to_constant_mask = {};
    }

    OPMetaInfo() = default;
};

class OPCache {
public:
    OPCache() : num_neighbours_to_cache(0), manager(MatchersManager()),
                m_ops_cache(std::vector<std::pair<std::shared_ptr<ngraph::Node>, OPMetaInfo>>()) {}

    static std::unique_ptr<OPCache> make_cache() {
        return std::unique_ptr<OPCache>(new OPCache());
    }

    void update_ops_cache(const std::shared_ptr<ngraph::Node> &op, const std::string &source_model = {});

    void update_ops_cache(const std::shared_ptr<ngraph::Function> &func, const std::string &source_model = {});

    void serialize_cached_ops(const std::string &serialization_dir);

    void set_num_neighbours_to_cache(size_t num) { num_neighbours_to_cache = num; }

protected:

    std::vector<std::pair<std::shared_ptr<ngraph::Node>, OPMetaInfo>> m_ops_cache;
    MatchersManager manager;
    size_t num_neighbours_to_cache = 0;
};
}  // namespace SubgraphsDumper
