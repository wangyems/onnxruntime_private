#include "gtest/gtest.h"

#error Stop

TEST(DummyTest, Passes) {
    ASSERT_TRUE(true);
    ASSERT_FALSE(true);
}
