#!/bin/bash
# Run this when acdc_chromium.rsf exists (after ACDC finishes on Chromium)
cd "$(dirname "$0")"
echo "=== ACDC Chromium TurboMQ & MoJo-FM ==="
java -jar turbomq.jar ../dataset/chromium/chromium-dependency.rsf acdc_chromium.rsf
java -jar mojo.jar acdc_chromium.rsf ../dataset/chromium/chromium-clustering.rsf -fm
