#!/usr/bin/env node

// JSON to RSF Converter - CLI Script
// Usage: node converter.js <json_file> <mapping_file> [output_file]

const fs = require("fs");
const path = require("path");

function parseMapping(mappingText) {
  const mapping = {};
  const lines = mappingText.trim().split("\n");

  lines.forEach((line) => {
    const match = line.match(/^(.+?)\s*->\s*(.+?)$/);
    if (match) {
      const nodeId = match[1].trim();
      const filename = match[2].trim();
      mapping[nodeId] = filename;
    }
  });

  return mapping;
}

function jsonToRsfWithMapping(jsonData, mappingText, useFilenames = true) {
  const rsfLines = [];
  const mapping = parseMapping(mappingText);
  const assignments = jsonData.assignments;

  if (!assignments || typeof assignments !== "object") {
    throw new Error('Invalid JSON structure. Expected "assignments" object.');
  }

  // Track statistics
  let mappedCount = 0;
  let unmappedCount = 0;

  Object.entries(assignments).forEach(([nodeId, clusterName]) => {
    let nodeName;

    if (useFilenames && mapping[nodeId]) {
      nodeName = mapping[nodeId];
      mappedCount++;
    } else {
      nodeName = nodeId;
      unmappedCount++;
    }

    rsfLines.push(`contain ${clusterName} ${nodeName}`);
  });

  return {
    output: rsfLines.join("\n"),
    stats: {
      totalNodes: Object.keys(assignments).length,
      mappedNodes: mappedCount,
      unmappedNodes: unmappedCount,
      clusters: [...new Set(Object.values(assignments))],
    },
  };
}

function main() {
  const args = process.argv.slice(2);

  if (args.length < 2) {
    console.log(
      "Usage: node converter.js <json_file> <mapping_file> [output_file]"
    );
    console.log("\nExample:");
    console.log(
      "  node converter.js qwen_code_480.json mapping_reverse.txt output.rsf"
    );
    process.exit(1);
  }

  const jsonFile = args[0];
  const mappingFile = args[1];
  const outputFile = args[2] || "output.rsf";

  try {
    // Read files
    if (!fs.existsSync(jsonFile)) {
      throw new Error(`JSON file not found: ${jsonFile}`);
    }
    if (!fs.existsSync(mappingFile)) {
      throw new Error(`Mapping file not found: ${mappingFile}`);
    }

    const jsonContent = fs.readFileSync(jsonFile, "utf8");
    const mappingContent = fs.readFileSync(mappingFile, "utf8");

    // Parse JSON
    const jsonData = JSON.parse(jsonContent);

    // Convert
    const result = jsonToRsfWithMapping(jsonData, mappingContent, true);

    // Write output
    fs.writeFileSync(outputFile, result.output, "utf8");

    // Display results
    console.log("\n✓ Conversion successful!");
    console.log(`\nStatistics:`);
    console.log(`  Total nodes: ${result.stats.totalNodes}`);
    console.log(`  Mapped nodes: ${result.stats.mappedCount}`);
    console.log(`  Unmapped nodes: ${result.stats.unmappedCount}`);
    console.log(`  Unique clusters: ${result.stats.clusters.length}`);
    console.log(`  Clusters: ${result.stats.clusters.join(", ")}`);
    console.log(`\nOutput written to: ${outputFile}`);
    console.log(`\nFirst 5 lines of output:`);
    console.log(result.output.split("\n").slice(0, 5).join("\n"));
  } catch (error) {
    console.error(`✗ Error: ${error.message}`);
    process.exit(1);
  }
}

main();
