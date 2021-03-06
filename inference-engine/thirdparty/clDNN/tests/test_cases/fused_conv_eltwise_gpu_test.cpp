// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "test_utils.h"

#include <cldnn/primitives/input_layout.hpp>
#include <cldnn/primitives/convolution.hpp>
#include <cldnn/primitives/eltwise.hpp>
#include <cldnn/primitives/reorder.hpp>
#include <cldnn/primitives/data.hpp>
#include <cldnn/primitives/depth_to_space.hpp>
#include <cldnn/primitives/fused_conv_eltwise.hpp>

#include <cassert>
#include <cmath>
#include <limits>

using namespace cldnn;
using namespace ::tests;
using namespace testing;

TEST(fused_conv_eltwise, basic_0)
{
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 4, 5 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });

    set_values(input, {
        1.0f,  2.0f, -15.f,  3.0f, 4.0f, -15.f, 5.0f,  6.0f, -15.f, 7.0f,
        -15.f, 0.0f,  0.0f, -15.f, 0.5f, -0.5f, -15.f, 8.0f,  1.5f,  5.2f
    });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        convolution("conv", "input", { "weights" }),
        eltwise("eltwise", "input", "conv", eltwise_mode::sum),
        reorder("out", "eltwise", format::bfyx, data_types::f32));

    build_options opt;
    opt.set_option(build_option::optimize_data(true));
    network network(engine, topology, opt);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output = outputs.begin()->second.get_memory();
    auto&& out_layout = output->get_layout();

    EXPECT_EQ(out_layout.format, format::bfyx);
    EXPECT_EQ(out_layout.size.batch[0], 1);
    EXPECT_EQ(out_layout.size.feature[0], 1);
    EXPECT_EQ(out_layout.size.spatial[0], 4);
    EXPECT_EQ(out_layout.size.spatial[1], 5);
}

TEST(fused_conv_eltwise, basic_image2d)
{
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 4, 128, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, { 1, 3, 256, 4 } });
    auto weights = engine.allocate_memory({ data_types::f16, format::bfyx, { 12, 4, 1, 1 } });

    auto input_data1 = generate_random_4d<FLOAT16>(1, 4, 2, 128, -1, 1);
    auto input_data1_bfyx = flatten_4d(format::bfyx, input_data1);
    set_values(input, input_data1_bfyx);

    auto input_data2 = generate_random_4d<FLOAT16>(1, 3, 4, 256, -1, 1);
    auto input_data2_bfyx = flatten_4d(format::bfyx, input_data2);
    set_values(input2, input_data2_bfyx);

    auto weights_data= generate_random_4d<FLOAT16>(12, 4, 1, 1, -1, 1);
    auto weights_data_bfyx = flatten_4d(format::bfyx, weights_data);
    set_values(weights, weights_data_bfyx);

    topology topology_act(
        input_layout("input", input->get_layout()),
        input_layout("input2", input2->get_layout()),
        data("weights", weights),
        convolution("conv", "input", { "weights" }),
        depth_to_space("depth_to_space", "conv", 2, depth_to_space_mode::blocks_first),
        eltwise("eltwise", "input2", "depth_to_space", eltwise_mode::sum)
    );

    build_options opt_act;
    opt_act.set_option(build_option::optimize_data(true));
    network network_act(engine, topology_act, opt_act);
    network_act.set_input_data("input", input);
    network_act.set_input_data("input2", input2);

    auto outputs_act = network_act.execute();
    EXPECT_EQ(outputs_act.size(), size_t(1));
    EXPECT_EQ(outputs_act.begin()->first, "eltwise");

    auto output_act = outputs_act.begin()->second.get_memory();
    cldnn::mem_lock<uint8_t> out_act_ptr(output_act, get_test_stream());

    topology topology_ref(
        input_layout("input", input->get_layout()),
        input_layout("input2", input2->get_layout()),
        data("weights", weights),
        convolution("conv", "input", { "weights" }),
        depth_to_space("depth_to_space", "conv", 2, depth_to_space_mode::blocks_first),
        eltwise("eltwise", "input2", "depth_to_space", eltwise_mode::sum),
        reorder("out", "eltwise", format::image_2d_rgba, data_types::u8));

    build_options opt_ref;
    opt_ref.set_option(build_option::optimize_data(false));
    network network_ref(engine, topology_ref, opt_ref);
    network_ref.set_input_data("input", input);
    network_ref.set_input_data("input2", input2);

    auto outputs_ref = network_ref.execute();
    EXPECT_EQ(outputs_ref.size(), size_t(1));
    EXPECT_EQ(outputs_ref.begin()->first, "out");

    auto output_ref = outputs_ref.begin()->second.get_memory();
    cldnn::mem_lock<uint8_t> out_ref_ptr(output_ref, get_test_stream());

    for (int i = 0;i < 3 * 256 * 4;i++) {
        EXPECT_EQ(out_act_ptr[i], out_ref_ptr[i]);
    }
}

TEST(fused_conv_eltwise, dont_fuse_if_conv_elt_are_outputs)
{
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 4, 5 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

    set_values(input, {
        1.0f,  2.0f, -15.f,  3.0f, 4.0f, -15.f, 5.0f,  6.0f, -15.f, 7.0f,
        -15.f, 0.0f,  0.0f, -15.f, 0.5f, -0.5f, -15.f, 8.0f,  1.5f,  5.2f
        });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        convolution("conv", "input", { "weights" }),
        eltwise("out", "input", "conv", eltwise_mode::sum));

    build_options opt;
    opt.set_option(build_option::optimize_data(true));
    network network(engine, topology, opt);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "out");

    auto output = outputs.begin()->second.get_memory();
    auto&& out_layout = output->get_layout();

    EXPECT_EQ(out_layout.format, format::bfyx);
    EXPECT_EQ(out_layout.size.batch[0], 1);
    EXPECT_EQ(out_layout.size.feature[0], 1);
    EXPECT_EQ(out_layout.size.spatial[0], 4);
    EXPECT_EQ(out_layout.size.spatial[1], 5);
}

template<typename InputTy,
         typename OutputTy>
class FusedConvTest : public testing::Test
{
protected:
    static constexpr bool is_pure_float = std::is_same<InputTy, float>::value;
    using OutputPreActivationTy = typename std::conditional<is_pure_float, float, int32_t>::type;
    using WeightsTy = typename std::conditional<is_pure_float, float, int8_t>::type;
    using BiasesTy = typename std::conditional<is_pure_float, float, int32_t>::type;

    topology the_topology;

    std::vector<InputTy> input_values;
    std::vector<WeightsTy> weights_values;
    std::vector<BiasesTy> biases_values;

    // Eltw part.
    std::vector<InputTy> non_conv_input_values;
    std::vector<OutputPreActivationTy> output_pre_relu;

    void add_feature(std::vector<InputTy> input,
                     std::vector<WeightsTy> weights,
                     BiasesTy bias,
                     std::vector<InputTy> non_conv_input,
                     std::vector<OutputPreActivationTy> output)
    {
        assert(non_conv_input.size() == output.size());
        input_values.insert(input_values.end(), input.begin(), input.end());
        weights_values.insert(
            weights_values.end(), weights.begin(), weights.end());
        biases_values.push_back(bias);
        non_conv_input_values.insert(non_conv_input_values.end(),
                                     non_conv_input.begin(),
                                     non_conv_input.end());
        output_pre_relu.insert(
            output_pre_relu.end(), output.begin(), output.end());
    }

    void do_test(const fused_conv_eltwise& fused_prim)
    {
        auto& engine = get_test_engine();

        int n_features = static_cast<int>(biases_values.size());

        auto input_shape = tensor(1, n_features, 4, 1);
        auto weights_shape = tensor(n_features, n_features, 3, 1);
        auto biases_shape = tensor(1, n_features, 1, 1);
        auto sum_input_shape = tensor(1, n_features, 2, 1);

        auto input = engine.allocate_memory({type_to_data_type<InputTy>::value, format::bfyx, input_shape});
        auto weights = engine.allocate_memory({type_to_data_type<WeightsTy>::value, format::bfyx, weights_shape});

        auto biases = engine.allocate_memory({type_to_data_type<BiasesTy>::value, format::bfyx, biases_shape});
        auto sum_input = engine.allocate_memory({type_to_data_type<InputTy>::value, format::bfyx, sum_input_shape});

        set_values(input, input_values);
        std::vector<WeightsTy> post_processed_weights_values(n_features
                                                             * n_features * 3);
        for (int output_feature = 0; output_feature < n_features; ++output_feature)
            for (int input_feature = 0; input_feature < n_features;
                 ++input_feature)
                for (int x = 0; x < 3; ++x)
                {
                    int idx =
                        output_feature * n_features * 3 + input_feature * 3 + x;
                    if (input_feature == output_feature)
                        post_processed_weights_values[idx] =
                            weights_values[input_feature * 3 + x];
                    else
                        post_processed_weights_values[idx] = 0;
                }
        set_values(weights, post_processed_weights_values);
        set_values(biases, biases_values);
        set_values(sum_input, non_conv_input_values);

        the_topology.add(input_layout("input", input->get_layout()));
        the_topology.add(data("weights", weights));
        the_topology.add(data("biases", biases));
        the_topology.add(data("sum_input", sum_input));
        the_topology.add(fused_prim);

        build_options opts;
        opts.set_option(build_option::optimize_data(false));

        network network(engine, the_topology, opts);
        network.set_input_data("input", input);

        auto outputs = network.execute();

        auto output_memory = outputs.at("fused_conv").get_memory();
        auto output_layout = output_memory->get_layout();
        cldnn::mem_lock<OutputTy> output_ptr(output_memory, get_test_stream());
        int y_size = output_layout.size.spatial[1];
        int x_size = output_layout.size.spatial[0];
        int f_size = output_layout.size.feature[0];
        int b_size = output_layout.size.batch[0];
        EXPECT_EQ(output_layout.format, format::bfyx);
        EXPECT_EQ(y_size, 1);
        EXPECT_EQ(x_size, 2);
        EXPECT_EQ(f_size, n_features);
        EXPECT_EQ(b_size, 1);

        for (int f = 0; f < f_size; f++)
            for (int x = 0; x < x_size; ++x)
            {
                // printf("f: %d, x: %d\n", f, x);
                OutputPreActivationTy expected =
                    pre_relu_to_output(output_pre_relu[f * x_size + x]);
                auto actual = static_cast<OutputPreActivationTy>(
                    output_ptr[f * x_size + x]);
                expect_eq(expected, actual);
            }
    }

private:
    template<typename T = OutputPreActivationTy>
    static typename std::enable_if<std::is_floating_point<T>::value>::type
    expect_eq(const OutputPreActivationTy& lhs, const OutputPreActivationTy& rhs)
    {
        EXPECT_NEAR(lhs, rhs, 0.001f);
    }

    template<typename T = OutputPreActivationTy>
    static typename std::enable_if<std::is_integral<T>::value>::type
    expect_eq(const OutputPreActivationTy& lhs, const OutputPreActivationTy& rhs)
    {
        EXPECT_EQ(lhs, rhs);
    }

    template <typename T>
    static T pre_relu_to_output(T pre_relu) {
      // No std::clamp before C++17 :(
      return std::min(
          static_cast<T>(std::numeric_limits<OutputTy>::max()),
          std::max(static_cast<T>(std::numeric_limits<OutputTy>::lowest()),
                   std::max(static_cast<T>(0), pre_relu)));
    }
};

class FusedConvTest_all_float : public FusedConvTest<float, float>
{};

TEST_F(FusedConvTest_all_float, DISABLED_basic) {
    add_feature({125.0f, 125.0f, 0.0f, 1.0f}, // input
                {2.0f, 0.0f, 1.0f},           // weights
                1.0f,                         // bias
                {-10.0f, -10.0f},             // non_conv_input
                {241.0f, 242.0f});            // output_pre_relu

    add_feature({125.0f, 125.0f, 0.0f, 1.0f}, // input
                {2.0f, 0.0f, 1.0f},           // weights
                0.0f,                         // bias
                {-10.0f, -11.0f},             // non_conv_input
                {480.0f, 480.0f});            // output_pre_relu

    do_test(fused_conv_eltwise("fused_conv",
                               "input",
                               "sum_input",
                               eltwise_mode::sum,
                               {"weights"},
                               {"biases"},
                               {{1, 1, 1, 1}}, // eltw_stride
                               {1, 1, 1, 1},   // stride
                               {0, 0, 0, 0},   // input_offset
                               {1, 1, 1, 1},   // dilation
                               false,          // conv_with_activation
                               0.0f,           // con_activation_slp
                               true,           // eltw_activation
                               0.0f));         // eltw_activation_slp
}
