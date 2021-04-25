def calculateObservability(levels, circuitDescription, testability):
	for level in reversed(levels):
		for lineInd in level:
			lineInfo = level[lineInd]
			observ = calculateLineObservability(lineInfo, lineInd, circuitDescription, testability)
			testability[lineInd]["obs"] = observ

def calculateLineObservability(lineInfo, lineInd, circuitDescription, testability):
    gate = lineInfo["entering"]
    if gate == None:
        obs = 0
        return obs
    else:
        gateType = circuitDescription[2][gate]["type"].lower()
        outputLine = circuitDescription[2][gate]["outputs"]
        if gateType == "and" or gateType == "nand":
            obs = testability[outputLine[0]]["obs"]
            for line in circuitDescription[2][gate]["inputs"]:
                if lineInd == line:
                    pass
                else:
                    obs = obs + testability[line]["control1"]

            obs = obs + 1
            return obs
        elif gateType == "or" or gateType == "nor":
            obs = testability[outputLine[0]]["obs"]
            for line in circuitDescription[2][gate]["inputs"]:
                if lineInd == line:
                    pass
                else:
                    obs = obs + testability[line]["control0"]
            obs = obs + 1
            return obs
        elif gateType == "xor":
            obs = testability[outputLine[0]]["obs"]
            for line in circuitDescription[2][gate]["inputs"]:
                if lineInd == line:
                    pass
                else:
                    obs += min(testability[line]["control0"], testability[line]["control1"])
            obs = obs + 1
            return obs
        elif gateType == "not":
            obs = testability[outputLine[0]]["obs"] + 1
            return obs
        elif gateType == "fanout":
            obs = testability[outputLine[0]]["obs"]
            for line in circuitDescription[2][gate]["outputs"]:
                obs = min(obs, testability[line]["obs"])
            return obs
        elif gateType == "buf" or gateType == "buff":
            obs = testability[outputLine[0]]["obs"]
            return obs
        else:
            print("error no gateType:" + gateType)
