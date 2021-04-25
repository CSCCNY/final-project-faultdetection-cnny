import pdb
import argparse
from operator import itemgetter
from create_level_array_from_circuit_description import (
    createLevelArrayFromCircuitDescription,
)
from parse_circuit_description_from_file import parseCircuitDescriptionFromFile
from calculate_controllability import calculateControllability
from calculate_observability import calculateObservability
from save_result_to_file import saveResultToFile


def calculateTestabilityMeasures(inputFileName, outputFileName):
    circuitDescription = parseCircuitDescriptionFromFile(inputFileName)
    levels = createLevelArrayFromCircuitDescription(circuitDescription)
    testability = {}

    calculateControllability(levels, circuitDescription, testability)
    calculateObservability(levels, circuitDescription, testability)

    for n in testability:
        testability[n]["testability s0"] = (
            testability[n]["control1"] + testability[n]["obs"]
        )
        testability[n]["testability s1"] = (
            testability[n]["control0"] + testability[n]["obs"]
        )

    sortedTest0 = sorted(
        testability, key=lambda x: (testability[x]["testability s0"],), reverse=True
    )
    sortedTest1 = sorted(
        testability, key=lambda x: (testability[x]["testability s1"],), reverse=True
    )

    for i, level in enumerate(levels):
        for l in level.keys():
            testability[l].update({"level": i})

    tenPercentSize = round(len(testability) * 0.1, 0)
    saveResultToFile(
        outputFileName,
        circuitDescription,
        testability,
        sortedTest0[0 : int(tenPercentSize)],
        sortedTest1[0 : int(tenPercentSize)],
    )
    print("Testability calculated, results saved to file " + outputFileName)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculates testability measures of a certain circuit"
    )
    parser.add_argument(
        "-i ",
        "--InputFileName",
        help="The name of the bench format file that describes the circuit",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o ",
        "--OutputFileName",
        help="The name of the file to export the result to",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    InputFileName = args.InputFileName
    OutputFileName = args.OutputFileName
    calculateTestabilityMeasures(InputFileName, OutputFileName)
