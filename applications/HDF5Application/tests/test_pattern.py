from typing import Any

import KratosMultiphysics.KratosUnittest as KratosUnittest
from KratosMultiphysics.HDF5Application.core.pattern import PatternEntity
from KratosMultiphysics.HDF5Application.core.pattern import GetMachingEntities

class TestGetMachingEntitiesString(KratosUnittest.TestCase):
    class StringPatternEntity(PatternEntity):
        def __init__(self,  name: str, current_item: Any) -> None:
            self.__name = name
            self.__item = current_item

        def Name(self) -> str:
            return self.__name

        def Iterate(self):
            for k, v in self.__item.items():
                yield TestGetMachingEntitiesString.StringPatternEntity(k, v)

        def IsLeaf(self) -> bool:
            return isinstance(self.__item, str)

        def Get(self) -> Any:
            return self.__item

    @staticmethod
    def __GenerateDictionaryFromStringList(list_of_strings: 'list[str]') -> dict:
        result = {}

        for input_str in list_of_strings:
            data = input_str.split("/")
            current_v = result
            if len(data) > 1:
                for v in data[:-1]:
                    if v not in current_v.keys():
                        current_v[v] = {}
                    current_v = current_v[v]

            current_v[data[-1]] = input_str

        return result

    @classmethod
    def setUpClass(cls) -> None:
        cls.data = TestGetMachingEntitiesString.__GenerateDictionaryFromStringList([
            "side_100_end/side_2/side_2.5_begin/side_61.h5",
            "side_10_end/side_3/side_3.5_begin/side_62.h5",
            "side_1_end/side_5/side_5.5_begin/side_63.h5",
            "side_2_end/side_6/side_6.5_begin/side_64.h5",
            "side_2_end/side_6/side_6_begin/side_65.h5",
            "side_2_end/side_3/side_4.6_begin/side_26.h5",
            "side_2_end/side_3/side_4.6_begin/side_26.0.h5",
            "side_20_end/side_60/side_2.8_begin/side_36.h5",
            "side_20_end/side_6/side_3.9_begin/side_56.h5",
            "side_2.5_end/side_5/side_8.5_begin/side_66.h5",
            "side_4_end/side_18/side_9.5_begin/side_86.h5",
        ])

    def testGetMachingEntitiesNonSorted(self):
        result = GetMachingEntities(
            TestGetMachingEntitiesString.StringPatternEntity("", self.data),
            "side_<TInt1>_end/side_<TInt2>/side_<TFloat1>_begin/side_<TInt3>.h5",
            {
                "<TInt1>"  : int,
                "<TInt2>"  : int,
                "<TInt3>"  : int,
                "<TFloat1>": float
            })
        self.assertEqual([
            "side_100_end/side_2/side_2.5_begin/side_61.h5",
            "side_10_end/side_3/side_3.5_begin/side_62.h5",
            "side_1_end/side_5/side_5.5_begin/side_63.h5",
            "side_2_end/side_6/side_6.5_begin/side_64.h5",
            "side_2_end/side_6/side_6_begin/side_65.h5",
            "side_2_end/side_3/side_4.6_begin/side_26.h5",
            "side_20_end/side_60/side_2.8_begin/side_36.h5",
            "side_20_end/side_6/side_3.9_begin/side_56.h5",
            "side_4_end/side_18/side_9.5_begin/side_86.h5"
        ], result)

    def testGetMachingEntitiesSorted(self):
        result = GetMachingEntities(
            TestGetMachingEntitiesString.StringPatternEntity("", self.data),
            "side_<TInt1>_end/side_<TInt2>/side_<TFloat1>_begin/side_<TInt3>.h5",
            {
                "<TInt1>"  : int,
                "<TInt2>"  : int,
                "<TInt3>"  : int,
                "<TFloat1>": float
            }, lambda _, *args: tuple(args))
        self.assertEqual([
            "side_1_end/side_5/side_5.5_begin/side_63.h5",
            "side_2_end/side_3/side_4.6_begin/side_26.h5",
            "side_2_end/side_6/side_6_begin/side_65.h5",
            "side_2_end/side_6/side_6.5_begin/side_64.h5",
            "side_4_end/side_18/side_9.5_begin/side_86.h5",
            "side_10_end/side_3/side_3.5_begin/side_62.h5",
            "side_20_end/side_6/side_3.9_begin/side_56.h5",
            "side_20_end/side_60/side_2.8_begin/side_36.h5",
            "side_100_end/side_2/side_2.5_begin/side_61.h5"
        ], result)

    def testGetMachingEntitiesCustomSorted(self):
        result = GetMachingEntities(
            TestGetMachingEntitiesString.StringPatternEntity("", self.data),
            "side_<TInt1>_end/side_<TInt2>/side_<TFloat1>_begin/side_<TInt3>.h5",
            {
                "<TInt1>"  : int,
                "<TInt2>"  : int,
                "<TInt3>"  : int,
                "<TFloat1>": float
            }, lambda _, i1, i2, f1, i3: tuple([f1, i2, i3, i1]))
        self.assertEqual([
            "side_100_end/side_2/side_2.5_begin/side_61.h5",
            "side_20_end/side_60/side_2.8_begin/side_36.h5",
            "side_10_end/side_3/side_3.5_begin/side_62.h5",
            "side_20_end/side_6/side_3.9_begin/side_56.h5",
            "side_2_end/side_3/side_4.6_begin/side_26.h5",
            "side_1_end/side_5/side_5.5_begin/side_63.h5",
            "side_2_end/side_6/side_6_begin/side_65.h5",
            "side_2_end/side_6/side_6.5_begin/side_64.h5",
            "side_4_end/side_18/side_9.5_begin/side_86.h5"
        ], result)

if __name__ == "__main__":
    KratosUnittest.main()