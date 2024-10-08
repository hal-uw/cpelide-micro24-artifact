/*
 * Copyright (c) 2021 Daniel R. Carvalho
 * All rights reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <cmath>

#include "base/gtest/cur_tick_fake.hh"
#include "base/stats/storage.hh"

// Instantiate the fake class to have a valid curTick of 0
GTestTickHandler tickHandler;

/** Increases the current tick by one. */
void increaseTick() { tickHandler.setCurTick(curTick() + 1); }

/** A pair of value and its number of samples, used for sampling. */
struct ValueSamples
{
    Stats::Counter value;
    Stats::Counter numSamples;

    ValueSamples(Stats::Counter value, Stats::Counter num_samples)
      : value(value), numSamples(num_samples)
    {
    }
};

/**
 * A mocked info class.
 * @todo There is no real dependency on the info class, so this must be
 * removed on a cleanup.
 */
class MockInfo : public Stats::Info
{
  public:
    MockInfo(Stats::StorageParams* storage_params)
      : Stats::Info()
    {
        this->storageParams = storage_params;
    }
    ~MockInfo() = default;

    bool check() const override { return true; }
    void prepare() override { }
    void reset() override { }
    bool zero() const override { return true; }
    void visit(Stats::Output &visitor) override { }
};

/** Test setting and getting a value to the storage. */
TEST(StatsStatStorTest, SetValueResult)
{
    Stats::StatStor stor(nullptr);
    Stats::Counter val;

    val = 10;
    stor.set(val);
    ASSERT_EQ(stor.value(), val);
    ASSERT_EQ(stor.result(), Stats::Result(val));

    val = 1234;
    stor.set(val);
    ASSERT_EQ(stor.value(), val);
    ASSERT_EQ(stor.result(), Stats::Result(val));
}

/** Test if prepare does not change the value. */
TEST(StatsStatStorTest, Prepare)
{
    Stats::StatStor stor(nullptr);
    Stats::Counter val;

    val = 10;
    stor.set(val);
    stor.prepare(nullptr);
    ASSERT_EQ(stor.value(), val);
    ASSERT_EQ(stor.result(), Stats::Result(val));
}

/** Test whether incrementing and decrementing work as expected. */
TEST(StatsStatStorTest, IncDec)
{
    Stats::StatStor stor(nullptr);
    Stats::Counter diff_val = 10;
    Stats::Counter val = 0;

    stor.inc(diff_val);
    val += diff_val;
    ASSERT_EQ(stor.value(), val);

    stor.inc(diff_val);
    val += diff_val;
    ASSERT_EQ(stor.value(), val);

    stor.dec(diff_val);
    val -= diff_val;
    ASSERT_EQ(stor.value(), val);

    stor.dec(diff_val);
    val -= diff_val;
    ASSERT_EQ(stor.value(), val);
}

/**
 * Test whether zero is correctly set as the reset value. The test order is
 * to check if it is initially zero on creation, then it is made non zero,
 * and finally reset to zero.
 */
TEST(StatsStatStorTest, ZeroReset)
{
    Stats::StatStor stor(nullptr);
    Stats::Counter val = 10;

    ASSERT_TRUE(stor.zero());

    stor.reset(nullptr);
    ASSERT_TRUE(stor.zero());

    stor.reset(nullptr);
    stor.inc(val);
    ASSERT_FALSE(stor.zero());
}

/** Test setting and getting a value to the storage. */
TEST(StatsAvgStorTest, SetValueResult)
{
    Stats::AvgStor stor(nullptr);
    Stats::Counter val;
    Stats::Result total = 0;
    Tick last_reset = 0;
    Tick last_tick = 0;

    val = 10;
    stor.set(val);
    last_tick = curTick();
    ASSERT_EQ(stor.value(), val);
    ASSERT_EQ(stor.result(), Stats::Result(total + val) /
        Stats::Result(curTick() - last_reset + 1));
    increaseTick();

    total += val * (curTick() - last_tick);
    val = 1234;
    stor.set(val);
    last_tick = curTick();
    ASSERT_EQ(stor.value(), val);
    ASSERT_EQ(stor.result(), Stats::Result(total + val) /
        Stats::Result(curTick() - last_reset + 1));
    increaseTick();
}

/**
 * Test whether getting the result in a different tick triggers an assertion.
 */
TEST(StatsAvgStorDeathTest, Result)
{
    Stats::AvgStor stor(nullptr);
    increaseTick();
    ASSERT_DEATH(stor.result(), ".+");
}

/**
 * Test whether getting the result in a different tick does not trigger an
 * assertion if storage is prepared.
 */
TEST(StatsAvgStorTest, Prepare)
{
    Stats::AvgStor stor(nullptr);
    Stats::Counter val = 10;
    Stats::Result total = 0;
    Tick last_reset = 0;
    Tick last_tick = 0;

    val = 10;
    stor.set(val);
    last_tick = curTick();
    ASSERT_EQ(stor.value(), val);
    ASSERT_EQ(stor.result(), Stats::Result(total + val) /
        Stats::Result(curTick() - last_reset + 1));
    increaseTick();

    total += val * (curTick() - last_tick);
    stor.prepare(nullptr);
    last_tick = curTick();
    ASSERT_EQ(stor.value(), val);
    ASSERT_EQ(stor.result(), Stats::Result(total + val) /
        Stats::Result(curTick() - last_reset + 1));
    increaseTick();
}

/** Test whether incrementing and decrementing work as expected. */
TEST(StatsAvgStorTest, IncDec)
{
    Stats::AvgStor stor(nullptr);
    Stats::Counter diff_val = 10;
    Stats::Counter val = 0;

    stor.set(diff_val);
    val += diff_val;
    ASSERT_EQ(stor.value(), val);

    stor.inc(diff_val);
    val += diff_val;
    ASSERT_EQ(stor.value(), val);

    stor.inc(diff_val);
    val += diff_val;
    ASSERT_EQ(stor.value(), val);

    stor.dec(diff_val);
    val -= diff_val;
    ASSERT_EQ(stor.value(), val);

    stor.dec(diff_val);
    val -= diff_val;
    ASSERT_EQ(stor.value(), val);
}

/**
 * Test whether zero is correctly set as the reset value. The test order is
 * to check if it is initially zero on creation, then it is made non zero,
 * and finally reset to zero.
 */
TEST(StatsAvgStorTest, ZeroReset)
{
    Stats::AvgStor stor(nullptr);
    Stats::Counter val = 10;

    ASSERT_TRUE(stor.zero());

    stor.reset(nullptr);
    ASSERT_TRUE(stor.zero());

    // Set current value to val, reset total and increase tick, so that the
    // next call to set will update the total to be different from zero
    stor.inc(val);
    stor.reset(nullptr);
    increaseTick();
    stor.inc(val);
    ASSERT_FALSE(stor.zero());
}

/**
 * Test that an assertion is thrown when no bucket size is provided before
 * sampling.
 */
TEST(StatsDistStorDeathTest, NoBucketSize)
{
    Stats::Counter val = 10;
    Stats::Counter num_samples = 5;
    Stats::DistStor::Params params;
    MockInfo info(&params);
    Stats::DistStor stor(&info);
    ASSERT_DEATH(stor.sample(val, num_samples), ".+");
}

/**
 * Test whether zero is correctly set as the reset value. The test order is
 * to check if it is initially zero on creation, then it is made non zero,
 * and finally reset to zero.
 */
TEST(StatsDistStorTest, ZeroReset)
{
    Stats::DistStor::Params params;
    params.bucket_size = 10;
    MockInfo info(&params);
    Stats::DistStor stor(&info);
    Stats::Counter val = 10;
    Stats::Counter num_samples = 5;

    ASSERT_TRUE(stor.zero());

    stor.reset(&info);
    stor.sample(val, num_samples);
    ASSERT_FALSE(stor.zero());

    stor.reset(&info);
    ASSERT_TRUE(stor.zero());
}

/**
 * Test that the size of this storage is equal to its counters vector's size,
 * and that after it has been set, nothing can modify it.
 */
TEST(StatsDistStorTest, Size)
{
    Stats::Counter val = 10;
    Stats::Counter num_samples = 5;
    Stats::Counter size = 20;
    Stats::DistData data;

    Stats::DistStor::Params params;
    params.bucket_size = 1;
    params.buckets = size;
    MockInfo info(&params);
    Stats::DistStor stor(&info);

    ASSERT_EQ(stor.size(), size);
    stor.sample(val, num_samples);
    ASSERT_EQ(stor.size(), size);
    stor.prepare(&info, data);
    ASSERT_EQ(stor.size(), size);
    stor.reset(&info);
    ASSERT_EQ(stor.size(), size);
    stor.zero();
    ASSERT_EQ(stor.size(), size);
}

/**
 * Compare both dist datas to see if their contents match.
 *
 * @param data The data being tested.
 * @param expected_data The ground truth.
 * @param no_log Whether log should not be compared.
 */
void
checkExpectedDistData(const Stats::DistData& data,
    const Stats::DistData& expected_data, bool no_log = true)
{
    ASSERT_EQ(data.type, expected_data.type);
    ASSERT_EQ(data.min, expected_data.min);
    ASSERT_EQ(data.max, expected_data.max);
    ASSERT_EQ(data.bucket_size, expected_data.bucket_size);
    ASSERT_EQ(data.min_val, expected_data.min_val);
    ASSERT_EQ(data.max_val, expected_data.max_val);
    ASSERT_EQ(data.sum, expected_data.sum);
    ASSERT_EQ(data.squares, expected_data.squares);
    if (!no_log) {
        ASSERT_EQ(data.logs, expected_data.logs);
    }
    ASSERT_EQ(data.samples, expected_data.samples);
    ASSERT_EQ(data.cvec.size(), expected_data.cvec.size());
    for (int i = 0; i < expected_data.cvec.size(); i++) {
        ASSERT_EQ(data.cvec[i], expected_data.cvec[i]);
    }
}

/**
 * Auxiliary function that finishes preparing the DistStor's expected values,
 * perform the calls to the storage's sample, and compares the expected data.
 *
 * @param params The params containing the number of buckets.
 * @param values The value-num_sample pairs to be sampled.
 * @param num_values Number of values in the values array.
 * @param expected_data Expected data after sampling, with the following values
 *  setup to the expected values: bucket_size, min, max_val, and cvec.
 */
void
prepareCheckDistStor(Stats::DistStor::Params& params, ValueSamples* values,
    int num_values, Stats::DistData& expected_data)
{
    MockInfo info(&params);
    Stats::DistStor stor(&info);

    Stats::Counter val;
    Stats::DistData data;

    expected_data.min = params.min;
    expected_data.max = params.max;
    expected_data.sum = 0;
    expected_data.squares = 0;
    expected_data.logs = 0;
    expected_data.samples = 0;

    // Populate storage with more data
    for (int i = 0; i < num_values; i++) {
        stor.sample(values[i].value, values[i].numSamples);

        val = values[i].value * values[i].numSamples;
        expected_data.sum += val;
        expected_data.squares += values[i].value * val;
        expected_data.samples += values[i].numSamples;
    }
    stor.prepare(&info, data);

    // DistStor does not use log
    checkExpectedDistData(data, expected_data, true);
}

/** Test setting and getting value from storage. */
TEST(StatsDistStorTest, SamplePrepareSingle)
{
    Stats::DistStor::Params params;
    params.min = 0;
    params.max = 99;
    params.bucket_size = 5;
    params.buckets = 20;

    ValueSamples values[] = {{10, 5}};
    int num_values = sizeof(values) / sizeof(ValueSamples);

    // Setup expected data
    Stats::DistData expected_data;
    expected_data.type = Stats::Dist;
    expected_data.bucket_size = params.bucket_size;
    expected_data.underflow = 0;
    expected_data.overflow = 0;
    expected_data.min_val = 10;
    expected_data.max_val = 10;
    expected_data.cvec.clear();
    expected_data.cvec.resize(params.buckets);
    expected_data.cvec[2] = 5;

    prepareCheckDistStor(params, values, num_values, expected_data);
}

/** Test setting and getting value from storage with multiple values. */
TEST(StatsDistStorTest, SamplePrepareMultiple)
{
    Stats::DistStor::Params params;
    params.min = 0;
    params.max = 99;
    params.bucket_size = 5;
    params.buckets = 20;

    // There are 20 buckets: [0,5[, [5,10[, [10,15[, ..., [95,100[.
    // We test that values that pass the maximum bucket value (1234, 12345678,
    // 100) are added to the overflow counter, and that the ones below the
    // minimum bucket value (-10, -1) are added to the underflow counter.
    // The extremes (0 and 99) are added to check if they go to the first and
    // last buckets.
    ValueSamples values[] = {{10, 5}, {1234, 2}, {12345678, 99}, {-10, 4},
        {17, 17}, {52, 63}, {18, 11}, {0, 1}, {99, 15}, {-1, 200}, {100, 50}};
    int num_values = sizeof(values) / sizeof(ValueSamples);

    // Setup variables that should always match params' values
    Stats::DistData expected_data;
    expected_data.type = Stats::Dist;
    expected_data.min_val = -10;
    expected_data.max_val = 12345678;
    expected_data.bucket_size = params.bucket_size;
    expected_data.underflow = 204;
    expected_data.overflow = 151;
    expected_data.sum = 0;
    expected_data.squares = 0;
    expected_data.samples = 0;
    expected_data.cvec.clear();
    expected_data.cvec.resize(params.buckets);
    expected_data.cvec[0] = 1;
    expected_data.cvec[2] = 5;
    expected_data.cvec[3] = 17+11;
    expected_data.cvec[10] = 63;
    expected_data.cvec[19] = 15;

    prepareCheckDistStor(params, values, num_values, expected_data);
}

/** Test resetting storage. */
TEST(StatsDistStorTest, Reset)
{
    Stats::DistStor::Params params;
    params.min = 0;
    params.max = 99;
    params.bucket_size = 5;
    params.buckets = 20;
    MockInfo info(&params);
    Stats::DistStor stor(&info);

    // Populate storage with random samples
    ValueSamples values[] = {{10, 5}, {1234, 2}, {12345678, 99}, {-10, 4},
        {17, 17}, {52, 63}, {18, 11}, {0, 1}, {99, 15}, {-1, 200}, {100, 50}};
    int num_values = sizeof(values) / sizeof(ValueSamples);
    for (int i = 0; i < num_values; i++) {
        stor.sample(values[i].value, values[i].numSamples);
    }

    // Reset storage, and make sure all data has been cleared
    stor.reset(&info);
    Stats::DistData data;
    stor.prepare(&info, data);

    Stats::DistData expected_data;
    expected_data.type = Stats::Dist;
    expected_data.bucket_size = params.bucket_size;
    expected_data.underflow = 0;
    expected_data.overflow = 0;
    expected_data.min = params.min;
    expected_data.max = params.max;
    expected_data.min_val = 0;
    expected_data.max_val = 0;
    expected_data.sum = 0;
    expected_data.squares = 0;
    expected_data.samples = 0;
    expected_data.cvec.clear();
    expected_data.cvec.resize(params.buckets);

    checkExpectedDistData(data, expected_data, true);
}

/**
 * Test that an assertion is thrown when no bucket size is provided before
 * sampling.
 */
TEST(StatsHistStorDeathTest, NoBucketSize)
{
    Stats::Counter val = 10;
    Stats::Counter num_samples = 5;

    // If no bucket size is specified, it is 0 by default
    Stats::HistStor::Params params;
    MockInfo info(&params);
    Stats::HistStor stor(&info);
    ASSERT_DEATH(stor.sample(val, num_samples), ".+");
}

/**
 * Test whether zero is correctly set as the reset value. The test order is
 * to check if it is initially zero on creation, then it is made non zero,
 * and finally reset to zero.
 */
TEST(StatsHistStorTest, ZeroReset)
{
    Stats::HistStor::Params params;
    params.buckets = 10;
    MockInfo info(&params);
    Stats::HistStor stor(&info);
    Stats::Counter val = 10;
    Stats::Counter num_samples = 5;

    ASSERT_TRUE(stor.zero());

    stor.reset(&info);
    stor.sample(val, num_samples);
    ASSERT_FALSE(stor.zero());

    stor.reset(&info);
    ASSERT_TRUE(stor.zero());
}

/**
 * Test that the size of this storage is equal to its counters vector's size,
 * and that after it has been set, nothing can modify it.
 */
TEST(StatsHistStorTest, Size)
{
    Stats::Counter val = 10;
    Stats::Counter num_samples = 5;
    Stats::DistData data;
    Stats::size_type sizes[] = {2, 10, 1234};

    // If no bucket size is specified, it is 0 by default
    {
        Stats::HistStor::Params params;
        MockInfo info(&params);
        Stats::HistStor stor(&info);

        ASSERT_EQ(stor.size(), 0);
        stor.prepare(&info, data);
        ASSERT_EQ(stor.size(), 0);
        stor.reset(&info);
        ASSERT_EQ(stor.size(), 0);
        stor.zero();
        ASSERT_EQ(stor.size(), 0);
    }

    for (int i = 0; i < (sizeof(sizes) / sizeof(Stats::size_type)); i++) {
        Stats::HistStor::Params params;
        params.buckets = sizes[i];
        MockInfo info(&params);
        Stats::HistStor stor(&info);

        ASSERT_EQ(stor.size(), sizes[i]);
        stor.sample(val, num_samples);
        ASSERT_EQ(stor.size(), sizes[i]);
        stor.prepare(&info, data);
        ASSERT_EQ(stor.size(), sizes[i]);
        stor.reset(&info);
        ASSERT_EQ(stor.size(), sizes[i]);
        stor.zero();
        ASSERT_EQ(stor.size(), sizes[i]);
    }
}

/**
 * Auxiliary function that finishes preparing the HistStor's expected values,
 * perform the calls to the storage's sample, and compares the expected data.
 *
 * @param params The params containing the number of buckets.
 * @param values The value-num_sample pairs to be sampled.
 * @param num_values Number of values in the values array.
 * @param expected_data Expected data after sampling, with the following values
 *  setup to the expected values: bucket_size, min, max_val, and cvec.
 */
void
prepareCheckHistStor(Stats::HistStor::Params& params, ValueSamples* values,
    int num_values, Stats::DistData& expected_data)
{
    MockInfo info(&params);
    Stats::HistStor stor(&info);

    Stats::Counter val;
    Stats::DistData data;
    bool no_log = false;

    expected_data.min_val = expected_data.min;
    expected_data.max = expected_data.max_val + expected_data.bucket_size - 1;
    expected_data.sum = 0;
    expected_data.squares = 0;
    expected_data.logs = 0;
    expected_data.samples = 0;

    // Populate storage with more data
    for (int i = 0; i < num_values; i++) {
        stor.sample(values[i].value, values[i].numSamples);

        val = values[i].value * values[i].numSamples;
        expected_data.sum += val;
        expected_data.squares += values[i].value * val;
        if (values[i].value < 0) {
            // Negative values don't have log, so mark log check to be skipped
            no_log = true;
        } else {
            expected_data.logs +=
                std::log(values[i].value) * values[i].numSamples;
        }
        expected_data.samples += values[i].numSamples;
    }
    stor.prepare(&info, data);
    checkExpectedDistData(data, expected_data, no_log);
}

/**
 * Test samples that fit in the initial buckets, and therefore do not need
 * to grow up.
 */
TEST(StatsHistStorTest, SamplePrepareFit)
{
    Stats::HistStor::Params params;
    params.buckets = 4;

    // Setup expected data for the hand-carved values given. The final buckets
    // will be divided at:
    //   Bkt0=[0,1[ , Bkt1=[1,2[, Bkt2=[2,3[, Bkt3=[3,4[
    ValueSamples values[] = {{0, 5}, {1, 2}, {2, 99}, {3, 4}};
    const int num_values = sizeof(values) / sizeof(ValueSamples);
    Stats::DistData expected_data;
    expected_data.type = Stats::Hist;
    expected_data.bucket_size = 1;
    expected_data.min = 0;
    expected_data.max_val = 3;
    expected_data.cvec.clear();
    expected_data.cvec.resize(params.buckets);
    expected_data.cvec[0] = 5;
    expected_data.cvec[1] = 2;
    expected_data.cvec[2] = 99;
    expected_data.cvec[3] = 4;

    prepareCheckHistStor(params, values, num_values, expected_data);
}

/**
 * Test samples that do not fit in the initial buckets, and therefore have
 * to grow up once.
 */
TEST(StatsHistStorTest, SamplePrepareSingleGrowUp)
{
    Stats::HistStor::Params params;
    params.buckets = 4;

    // Setup expected data for the hand-carved values given. Since there
    // are four buckets, and the highest value is 4, the bucket size will
    // grow to be 2. The final buckets will be divided at:
    //   Bkt0=[0,2[ , Bkt1=[2,4[, Bkt2=[4,6[, Bkt3=[6,8[
    ValueSamples values[] = {{0, 5}, {1, 2}, {2, 99}, {4, 4}};
    const int num_values = sizeof(values) / sizeof(ValueSamples);
    Stats::DistData expected_data;
    expected_data.type = Stats::Hist;
    expected_data.bucket_size = 2;
    expected_data.min = 0;
    expected_data.max_val = 6;
    expected_data.cvec.clear();
    expected_data.cvec.resize(params.buckets);
    expected_data.cvec[0] = 5+2;
    expected_data.cvec[1] = 99;
    expected_data.cvec[2] = 4;
    expected_data.cvec[3] = 0;

    prepareCheckHistStor(params, values, num_values, expected_data);
}

/**
 * Test samples that do not fit in the initial buckets, and therefore have
 * to grow up a few times.
 */
TEST(StatsHistStorTest, SamplePrepareMultipleGrowUp)
{
    Stats::HistStor::Params params;
    params.buckets = 4;

    // Setup expected data for the hand-carved values given. Since there
    // are four buckets, and the highest value is 4, the bucket size will
    // grow thrice to become 8. The final buckets will be divided at:
    //   Bkt0=[0,8[ , Bkt1=[8,16[, Bkt2=[16,24[, Bkt3=[24,32[
    ValueSamples values[] = {{0, 5}, {1, 2}, {2, 99}, {16, 4}};
    const int num_values = sizeof(values) / sizeof(ValueSamples);
    Stats::DistData expected_data;
    expected_data.type = Stats::Hist;
    expected_data.bucket_size = 8;
    expected_data.min = 0;
    expected_data.max_val = 24;
    expected_data.cvec.clear();
    expected_data.cvec.resize(params.buckets);
    expected_data.cvec[0] = 5+2+99;
    expected_data.cvec[1] = 0;
    expected_data.cvec[2] = 4;
    expected_data.cvec[3] = 0;

    prepareCheckHistStor(params, values, num_values, expected_data);
}

/**
 * Test samples that have a negative value, and therefore do not fit in the
 * initial buckets. Since this involves using negative values, the logs
 * become irrelevant.
 */
TEST(StatsHistStorTest, SamplePrepareGrowDownOddBuckets)
{
    Stats::HistStor::Params params;
    params.buckets = 5;

    // Setup expected data for the hand-carved values given. Since there
    // is a negative value, the min bucket will change, and the bucket size
    // will grow to be 2. The final buckets will be divided at:
    //   Bkt0=[-4,-2[ , Bkt1=[-2,-0[, Bkt2=[0,2[, Bkt3=[2,4[, Bkt4=[4,6[
    ValueSamples values[] =
        {{0, 5}, {1, 2}, {2, 99}, {3, 12}, {4, 33}, {-1, 4}};
    const int num_values = sizeof(values) / sizeof(ValueSamples);
    Stats::DistData expected_data;
    expected_data.type = Stats::Hist;
    expected_data.bucket_size = 2;
    expected_data.min = -4;
    expected_data.max_val = 4;
    expected_data.cvec.clear();
    expected_data.cvec.resize(params.buckets);
    expected_data.cvec[0] = 0;
    expected_data.cvec[1] = 4;
    expected_data.cvec[2] = 5+2;
    expected_data.cvec[3] = 99+12;
    expected_data.cvec[4] = 33;

    prepareCheckHistStor(params, values, num_values, expected_data);
}

/**
 * Test samples that have a negative value, and therefore do not fit in the
 * initial buckets. Since this involves using negative values, the logs
 * become irrelevant.
 */
TEST(StatsHistStorTest, SamplePrepareGrowDownEvenBuckets)
{
    Stats::HistStor::Params params;
    params.buckets = 4;

    // Setup expected data for the hand-carved values given. Since there
    // is a negative value, the min bucket will change, and the bucket size
    // will grow to be 2. The final buckets will be divided at:
    //   Bkt0=[-4,-2[ , Bkt1=[-2,0[, Bkt2=[0,2[, Bkt3=[2,4[
    ValueSamples values[] = {{0, 5}, {1, 2}, {2, 99}, {-1, 4}};
    const int num_values = sizeof(values) / sizeof(ValueSamples);
    Stats::DistData expected_data;
    expected_data.type = Stats::Hist;
    expected_data.bucket_size = 2;
    expected_data.min = -4;
    expected_data.max_val = 2;
    expected_data.cvec.clear();
    expected_data.cvec.resize(params.buckets);
    expected_data.cvec[0] = 0;
    expected_data.cvec[1] = 4;
    expected_data.cvec[2] = 5+2;
    expected_data.cvec[3] = 99;

    prepareCheckHistStor(params, values, num_values, expected_data);
}

/**
 * Test samples that have one low negative value, and therefore do not fit
 * in the initial buckets and have to grow down a few times. Since this
 * involves using negative values, the logs become irrelevant.
 */
TEST(StatsHistStorTest, SamplePrepareGrowDownGrowOutOddBuckets)
{
    Stats::HistStor::Params params;
    params.buckets = 5;

    // Setup expected data for the hand-carved values given. Since there
    // is a negative value, the min bucket will change, and the bucket size
    // will grow to be 8. The final buckets will be divided at:
    //   Bkt0=[-16,-8[ , Bkt1=[-8,0[, Bkt2=[0,8[, Bkt3=[8,16[, Bkt4=[16,24[
    ValueSamples values[] =
        {{0, 5}, {1, 2}, {2, 99}, {3, 12}, {4, 33}, {-12, 4}};
    const int num_values = sizeof(values) / sizeof(ValueSamples);
    Stats::DistData expected_data;
    expected_data.type = Stats::Hist;
    expected_data.bucket_size = 8;
    expected_data.min = -16;
    expected_data.max_val = 16;
    expected_data.cvec.clear();
    expected_data.cvec.resize(params.buckets);
    expected_data.cvec[0] = 4;
    expected_data.cvec[1] = 0;
    expected_data.cvec[2] = 5+2+99+12+33;
    expected_data.cvec[3] = 0;
    expected_data.cvec[4] = 0;

    prepareCheckHistStor(params, values, num_values, expected_data);
}

/**
 * Test samples that have one low negative value, and therefore do not fit
 * in the initial buckets and have to grow down a few times. Since this
 * involves using negative values, the logs become irrelevant.
 */
TEST(StatsHistStorTest, SamplePrepareGrowDownGrowOutEvenBuckets)
{
    Stats::HistStor::Params params;
    params.buckets = 4;

    // Setup expected data for the hand-carved values given. Since there
    // is a negative value, the min bucket will change, and the bucket size
    // will grow to be 8. The final buckets will be divided at:
    //   Bkt0=[-16,-8[ , Bkt1=[-8,0[, Bkt2=[0,8[, Bkt3=[8,16[
    ValueSamples values[] =
        {{0, 5}, {1, 2}, {2, 99}, {3, 12}, {-12, 4}};
    const int num_values = sizeof(values) / sizeof(ValueSamples);
    Stats::DistData expected_data;
    expected_data.type = Stats::Hist;
    expected_data.bucket_size = 8;
    expected_data.min = -16;
    expected_data.max_val = 8;
    expected_data.cvec.clear();
    expected_data.cvec.resize(params.buckets);
    expected_data.cvec[0] = 4;
    expected_data.cvec[1] = 0;
    expected_data.cvec[2] = 5+2+99+12;
    expected_data.cvec[3] = 0;

    prepareCheckHistStor(params, values, num_values, expected_data);
}

/**
 * Test a complex sample set with negative values, and therefore multiple
 * grows will happen. Since this involves using negative values, the logs
 * become irrelevant.
 */
TEST(StatsHistStorTest, SamplePrepareMultipleGrowOddBuckets)
{
    Stats::HistStor::Params params;
    params.buckets = 5;

    // Setup expected data for the hand-carved values given. This adds quite
    // a few positive and negative samples, and the bucket size will grow to
    // be 64. The final buckets will be divided at:
    //   Bkt0=[-128,-64[ , Bkt1=[-64,0[, Bkt2=[0,64[, Bkt3=[64,128[,
    //   Bkt4=[128,192[
    ValueSamples values[] =
        {{0, 5}, {7, 2}, {31, 99}, {-8, 12}, {127, 4}, {-120, 53}, {-50, 1}};
    const int num_values = sizeof(values) / sizeof(ValueSamples);
    Stats::DistData expected_data;
    expected_data.type = Stats::Hist;
    expected_data.bucket_size = 64;
    expected_data.min = -128;
    expected_data.max_val = 128;
    expected_data.cvec.clear();
    expected_data.cvec.resize(params.buckets);
    expected_data.cvec[0] = 53;
    expected_data.cvec[1] = 12+1;
    expected_data.cvec[2] = 5+2+99;
    expected_data.cvec[3] = 4;
    expected_data.cvec[4] = 0;

    prepareCheckHistStor(params, values, num_values, expected_data);
}

/**
 * Test a complex sample set with negative values, and therefore multiple
 * grows will happen. Since this involves using negative values, the logs
 * become irrelevant.
 */
TEST(StatsHistStorTest, SamplePrepareMultipleGrowEvenBuckets)
{
    Stats::HistStor::Params params;
    params.buckets = 4;

    // Setup expected data for the hand-carved values given. This adds quite
    // a few positive and negative samples, and the bucket size will grow to
    // be 64. The final buckets will be divided at:
    //   Bkt0=[-128,-64[ , Bkt1=[-64,0[, Bkt2=[0,64[, Bkt3=[64,128[
    ValueSamples values[] =
        {{0, 5}, {7, 2}, {31, 99}, {-8, 12}, {127, 4}, {-120, 53}, {-50, 1}};
    const int num_values = sizeof(values) / sizeof(ValueSamples);
    Stats::DistData expected_data;
    expected_data.type = Stats::Hist;
    expected_data.bucket_size = 64;
    expected_data.min = -128;
    expected_data.max_val = 64;
    expected_data.cvec.clear();
    expected_data.cvec.resize(params.buckets);
    expected_data.cvec[0] = 53;
    expected_data.cvec[1] = 12+1;
    expected_data.cvec[2] = 5+2+99;
    expected_data.cvec[3] = 4;

    prepareCheckHistStor(params, values, num_values, expected_data);
}

/** Test resetting storage. */
TEST(StatsHistStorTest, Reset)
{
    Stats::HistStor::Params params;
    params.buckets = 4;
    MockInfo info(&params);
    Stats::HistStor stor(&info);

    // Setup expected data for the hand-carved values given. This adds quite
    // a few positive and negative samples, and the bucket size will grow to
    // be 64. The final buckets will be divided at:
    //   Bkt0=[-128,-64[ , Bkt1=[-64,0[, Bkt2=[0,64[, Bkt3=[64,128[
    ValueSamples values[] =
        {{0, 5}, {7, 2}, {31, 99}, {-8, 12}, {127, 4}, {-120, 53}, {-50, 1}};
    const int num_values = sizeof(values) / sizeof(ValueSamples);
    for (int i = 0; i < num_values; i++) {
        stor.sample(values[i].value, values[i].numSamples);
    }

    // Reset storage, and make sure all data has been cleared:
    //   Bkt0=[0,1[ , Bkt1=[1,2[, Bkt2=[2,3[, Bkt3=[3,4[
    stor.reset(&info);
    Stats::DistData expected_data;
    expected_data.type = Stats::Hist;
    expected_data.bucket_size = 1;
    expected_data.min = 0;
    expected_data.max_val = 3;
    expected_data.cvec.clear();
    expected_data.cvec.resize(params.buckets);
    prepareCheckHistStor(params, values, 0, expected_data);
}

/** Test whether adding storages with different sizes triggers an assertion. */
TEST(StatsHistStorDeathTest, AddDifferentSize)
{
    Stats::HistStor::Params params;
    params.buckets = 4;
    MockInfo info(&params);
    Stats::HistStor stor(&info);

    Stats::HistStor::Params params2;
    params2.buckets = 5;
    MockInfo info2(&params2);
    Stats::HistStor stor2(&info2);

    ASSERT_DEATH(stor.add(&stor2), ".+");
}

/** Test whether adding storages with different min triggers an assertion. */
TEST(StatsHistStorDeathTest, AddDifferentMin)
{
    Stats::HistStor::Params params;
    params.buckets = 4;
    MockInfo info(&params);
    Stats::HistStor stor(&info);
    stor.sample(-1, 3);

    // On creation, the storage's min is zero
    Stats::HistStor::Params params2;
    params2.buckets = 4;
    MockInfo info2(&params2);
    Stats::HistStor stor2(&info2);

    ASSERT_DEATH(stor.add(&stor2), ".+");
}

/** Test merging two histograms. */
TEST(StatsHistStorTest, Add)
{
    Stats::HistStor::Params params;
    params.buckets = 4;
    MockInfo info(&params);

    // Setup first storage. Buckets are:
    //   Bkt0=[0,16[, Bkt1=[16,32[, Bkt2=[32,48[, Bkt3=[58,64[
    Stats::HistStor stor(&info);
    ValueSamples values[] = {{0, 5}, {3, 2}, {20, 37}, {32, 18}};
    int num_values = sizeof(values) / sizeof(ValueSamples);
    for (int i = 0; i < num_values; i++) {
        stor.sample(values[i].value, values[i].numSamples);
    }
    Stats::DistData data;
    stor.prepare(&info, data);

    // Setup second storage. Buckets are:
    //   Bkt0=[0,32[, Bkt1=[32,64[, Bkt2=[64,96[, Bkt3=[96,128[
    Stats::HistStor stor2(&info);
    ValueSamples values2[] = {{10, 10}, {0, 1}, {80, 4}, {17, 100}, {95, 79}};
    int num_values2 = sizeof(values2) / sizeof(ValueSamples);
    for (int i = 0; i < num_values2; i++) {
        stor2.sample(values2[i].value, values2[i].numSamples);
    }
    Stats::DistData data2;
    stor2.prepare(&info, data2);

    // Perform the merge
    stor.add(&stor2);
    Stats::DistData merge_data;
    stor.prepare(&info, merge_data);

    // Setup expected data. Buckets are:
    //   Bkt0=[0,32[, Bkt1=[32,64[, Bkt2=[64,96[, Bkt3=[96,128[
    Stats::DistData expected_data;
    expected_data.type = Stats::Hist;
    expected_data.bucket_size = 32;
    expected_data.min = 0;
    expected_data.max = 127;
    expected_data.min_val = 0;
    expected_data.max_val = 96;
    expected_data.cvec.clear();
    expected_data.cvec.resize(params.buckets);
    expected_data.cvec[0] = 5+2+37+10+1+100;
    expected_data.cvec[1] = 18;
    expected_data.cvec[2] = 4+79;
    expected_data.cvec[3] = 0;
    expected_data.sum = data.sum + data2.sum;
    expected_data.squares = data.squares + data2.squares;
    expected_data.logs = data.squares + data2.logs;
    expected_data.samples = data.samples + data2.samples;

    // Compare results
    checkExpectedDistData(merge_data, expected_data, false);
}

/**
 * Test whether zero is correctly set as the reset value. The test order is
 * to check if it is initially zero on creation, then it is made non zero,
 * and finally reset to zero.
 */
TEST(StatsSampleStorTest, ZeroReset)
{
    Stats::SampleStor stor(nullptr);
    Stats::Counter val = 10;
    Stats::Counter num_samples = 5;

    ASSERT_TRUE(stor.zero());

    stor.reset(nullptr);
    stor.sample(val, num_samples);
    ASSERT_FALSE(stor.zero());

    stor.reset(nullptr);
    ASSERT_TRUE(stor.zero());
}

/** Test setting and getting value from storage. */
TEST(StatsSampleStorTest, SamplePrepare)
{
    Stats::SampleStor stor(nullptr);
    ValueSamples values[] = {{10, 5}, {1234, 2}, {0xFFFFFFFF, 18}};
    int num_values = sizeof(values) / sizeof(ValueSamples);
    Stats::Counter val;
    Stats::DistData data;
    Stats::DistData expected_data;
    Stats::DistParams params(Stats::Deviation);
    MockInfo info(&params);

    // Simple test with one value being sampled
    stor.sample(values[0].value, values[0].numSamples);
    stor.prepare(&info, data);
    val = values[0].value * values[0].numSamples;
    expected_data.type = Stats::Deviation;
    expected_data.sum = val;
    expected_data.squares = values[0].value * val;
    expected_data.samples = values[0].numSamples;
    ASSERT_EQ(data.type, expected_data.type);
    ASSERT_EQ(data.sum, expected_data.sum);
    ASSERT_EQ(data.squares, expected_data.squares);
    ASSERT_EQ(data.samples, expected_data.samples);

    // Reset storage, and make sure all data has been cleared
    expected_data.sum = 0;
    expected_data.squares = 0;
    expected_data.samples = 0;
    stor.reset(nullptr);
    stor.prepare(&info, data);
    ASSERT_EQ(data.type, expected_data.type);
    ASSERT_EQ(data.sum, expected_data.sum);
    ASSERT_EQ(data.squares, expected_data.squares);
    ASSERT_EQ(data.samples, expected_data.samples);

    // Populate storage with more data
    for (int i = 0; i < num_values; i++) {
        stor.sample(values[i].value, values[i].numSamples);

        val = values[i].value * values[i].numSamples;
        expected_data.sum += val;
        expected_data.squares += values[i].value * val;
        expected_data.samples += values[i].numSamples;
    }
    stor.prepare(&info, data);
    ASSERT_EQ(data.type, expected_data.type);
    ASSERT_EQ(data.sum, expected_data.sum);
    ASSERT_EQ(data.squares, expected_data.squares);
    ASSERT_EQ(data.samples, expected_data.samples);
}

/** The size is always 1, no matter which functions have been called. */
TEST(StatsSampleStorTest, Size)
{
    Stats::SampleStor stor(nullptr);
    Stats::Counter val = 10;
    Stats::Counter num_samples = 5;
    Stats::DistData data;
    Stats::DistParams params(Stats::Deviation);
    MockInfo info(&params);

    ASSERT_EQ(stor.size(), 1);
    stor.sample(val, num_samples);
    ASSERT_EQ(stor.size(), 1);
    stor.prepare(&info, data);
    ASSERT_EQ(stor.size(), 1);
    stor.reset(nullptr);
    ASSERT_EQ(stor.size(), 1);
    stor.zero();
    ASSERT_EQ(stor.size(), 1);
}

/**
 * Test whether zero is correctly set as the reset value. The test order is
 * to check if it is initially zero on creation, then it is made non zero,
 * and finally reset to zero.
 */
TEST(StatsAvgSampleStorTest, ZeroReset)
{
    Stats::AvgSampleStor stor(nullptr);
    Stats::Counter val = 10;
    Stats::Counter num_samples = 5;

    ASSERT_TRUE(stor.zero());

    stor.reset(nullptr);
    stor.sample(val, num_samples);
    ASSERT_FALSE(stor.zero());

    stor.reset(nullptr);
    ASSERT_TRUE(stor.zero());
}

/** Test setting and getting value from storage. */
TEST(StatsAvgSampleStorTest, SamplePrepare)
{
    Stats::AvgSampleStor stor(nullptr);
    ValueSamples values[] = {{10, 5}, {1234, 2}, {0xFFFFFFFF, 18}};
    int num_values = sizeof(values) / sizeof(ValueSamples);
    Stats::Counter val;
    Stats::DistData data;
    Stats::DistData expected_data;
    Stats::DistParams params(Stats::Deviation);
    MockInfo info(&params);

    // Simple test with one value being sampled
    stor.sample(values[0].value, values[0].numSamples);
    stor.prepare(&info, data);
    val = values[0].value * values[0].numSamples;
    expected_data.type = Stats::Deviation;
    expected_data.sum = val;
    expected_data.squares = values[0].value * val;
    ASSERT_EQ(data.type, expected_data.type);
    ASSERT_EQ(data.sum, expected_data.sum);
    ASSERT_EQ(data.squares, expected_data.squares);
    ASSERT_EQ(data.samples, curTick());

    increaseTick();

    // Reset storage, and make sure all data has been cleared
    expected_data.sum = 0;
    expected_data.squares = 0;
    stor.reset(nullptr);
    stor.prepare(&info, data);
    ASSERT_EQ(data.type, expected_data.type);
    ASSERT_EQ(data.sum, expected_data.sum);
    ASSERT_EQ(data.squares, expected_data.squares);
    ASSERT_EQ(data.samples, curTick());

    increaseTick();

    // Populate storage with more data
    for (int i = 0; i < num_values; i++) {
        stor.sample(values[i].value, values[i].numSamples);

        val = values[i].value * values[i].numSamples;
        expected_data.sum += val;
        expected_data.squares += values[i].value * val;
    }
    stor.prepare(&info, data);
    ASSERT_EQ(data.type, expected_data.type);
    ASSERT_EQ(data.sum, expected_data.sum);
    ASSERT_EQ(data.squares, expected_data.squares);
    ASSERT_EQ(data.samples, curTick());
}

/** The size is always 1, no matter which functions have been called. */
TEST(StatsAvgSampleStorTest, Size)
{
    Stats::AvgSampleStor stor(nullptr);
    Stats::Counter val = 10;
    Stats::Counter num_samples = 5;
    Stats::DistData data;
    Stats::DistParams params(Stats::Deviation);
    MockInfo info(&params);

    ASSERT_EQ(stor.size(), 1);
    stor.sample(val, num_samples);
    ASSERT_EQ(stor.size(), 1);
    stor.prepare(&info, data);
    ASSERT_EQ(stor.size(), 1);
    stor.reset(nullptr);
    ASSERT_EQ(stor.size(), 1);
    stor.zero();
    ASSERT_EQ(stor.size(), 1);
}

/**
 * Test whether zero is correctly set as the reset value. The test order is
 * to check if it is initially zero on creation, then it is made non zero,
 * and finally reset to zero.
 */
TEST(StatsSparseHistStorTest, ZeroReset)
{
    Stats::SparseHistStor stor(nullptr);
    Stats::Counter val = 10;
    Stats::Counter num_samples = 5;

    ASSERT_TRUE(stor.zero());

    stor.reset(nullptr);
    stor.sample(val, num_samples);
    ASSERT_FALSE(stor.zero());

    stor.reset(nullptr);
    ASSERT_TRUE(stor.zero());
}

/** Test setting and getting value from storage. */
TEST(StatsSparseHistStorTest, SamplePrepare)
{
    Stats::SparseHistStor stor(nullptr);
    ValueSamples values[] = {{10, 5}, {1234, 2}, {0xFFFFFFFF, 18}};
    int num_values = sizeof(values) / sizeof(ValueSamples);
    Stats::Counter total_samples;
    Stats::SparseHistData data;

    // Simple test with one value being sampled
    stor.sample(values[0].value, values[0].numSamples);
    stor.prepare(nullptr, data);
    ASSERT_EQ(stor.size(), 1);
    ASSERT_EQ(data.cmap.size(), 1);
    ASSERT_EQ(data.cmap[values[0].value], values[0].numSamples);
    ASSERT_EQ(data.samples, values[0].numSamples);

    // Reset storage, and make sure all data has been cleared
    stor.reset(nullptr);
    stor.prepare(nullptr, data);
    ASSERT_EQ(stor.size(), 0);
    ASSERT_EQ(data.cmap.size(), 0);
    ASSERT_EQ(data.samples, 0);

    // Populate storage with more data
    for (int i = 0; i < num_values; i++) {
        stor.sample(values[i].value, values[i].numSamples);
    }
    stor.prepare(nullptr, data);
    total_samples = 0;
    ASSERT_EQ(stor.size(), num_values);
    ASSERT_EQ(data.cmap.size(), num_values);
    for (int i = 0; i < num_values; i++) {
        ASSERT_EQ(data.cmap[values[i].value], values[i].numSamples);
        total_samples += values[i].numSamples;
    }
    ASSERT_EQ(data.samples, total_samples);
}
