document.addEventListener("DOMContentLoaded", function () {
    const edaData = JSON.parse(localStorage.getItem("edaData"));
    const reportContainer = document.getElementById("reportContainer");

    if (edaData) {
        // Helper function to generate rows with a subtitle column
        function generateDataRow(subtitle, dataObject) {
            return `
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">${subtitle}</td>
                    ${Object.values(dataObject).map(value => `<td style="border: 1px solid #ddd; padding: 8px;">${value ?? 'N/A'}</td>`).join("")}
                </tr>
            `;
        }

        // Creating headers dynamically based on numeric summary keys
        const numericSummaryKeys = edaData.descriptive_statistics?.numeric_summary 
            ? Object.keys(edaData.descriptive_statistics.numeric_summary.age) 
            : [];

        const tableHeaderRow = `
            <tr>
                <th style="border: 1px solid #ddd; padding: 8px;">Metric</th>
                ${numericSummaryKeys.map(key => `<th style="border: 1px solid #ddd; padding: 8px;">${key}</th>`).join("")}
            </tr>
        `;

        const tableHeaderRow1 = edaData.dataset_overview?.columns ? `
        <tr>
            <th style="border: 1px solid #ddd; padding: 8px;">Metric</th>
            ${Object.keys(edaData.dataset_overview.columns).map(key => `<th style="border: 1px solid #ddd; padding: 8px;">${key}</th>`).join("")}
        </tr>
    ` : '';

        // Dataset Overview Section
        const datasetOverviewTable = `
            <table style="width: 100%; border-collapse: collapse; margin-top: 8px;">
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">Rows</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">${edaData.dataset_overview.shape[0]}</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">Columns</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">${edaData.dataset_overview.shape[1]}</td>
                </tr>
            </table>
            <table style="width: 100%; border-collapse: collapse; margin-top: 8px;">
                    ${tableHeaderRow1}
                    ${generateDataRow("Columns", edaData.dataset_overview?.columns)}
                    ${generateDataRow("Missing Values", edaData.dataset_overview?.missing_values)}
                    ${generateDataRow("Unique Values", edaData.dataset_overview?.unique_values)}
                </table>
        `;

        // Descriptive Statistics Section
        const descriptiveStatisticsTable = `
            <table style="width: 100%; border-collapse: collapse; margin-top: 8px;">
                ${tableHeaderRow}
                ${Object.entries(edaData.descriptive_statistics.numeric_summary).map(
                    ([metric, values]) => generateDataRow(metric, values)
                ).join("")}
            </table>
        `;

        // Data Quality Analysis Section
        const dataQualityAnalysisHTML = `
            <div class="card">
                <h3>Data Quality Analysis</h3>
                <p><strong>Duplicates Count:</strong> ${edaData.data_quality_analysis.duplicates_count}</p>
                
                <h4>Missing Data Summary (Percentage)</h4>
                <table style="width: 100%; border-collapse: collapse;">
                    ${Object.entries(edaData.data_quality_analysis.missing_data_summary)
                        .map(([column, percent]) => `
                            <tr>
                                <th style="border: 1px solid #ddd; padding: 8px;">${column}</th>
                                <td style="border: 1px solid #ddd; padding: 8px;">${percent}%</td>
                            </tr>
                        `)
                        .join("")}
                </table>
                
                ${
                    edaData.data_quality_analysis.missing_data_pattern
                        ? `<img src="data:image/png;base64,${edaData.data_quality_analysis.missing_data_pattern}" alt="Missing Data Pattern">`
                        : "<p>No missing data to visualize.</p>"
                }
            </div>
        `;

        // Univariate Analysis Section
        const univariateAnalysisHTML = `
            <div class="card">
                <h3>Univariate Analysis</h3>
                ${Object.keys(edaData.univariate_analysis.histograms)
                    .map((col) => `
                        <h4>${col} Histogram</h4>
                        <img src="data:image/png;base64,${edaData.univariate_analysis.histograms[col]}" alt="${col} Histogram">
                    `)
                    .join("")}
            </div>
        `;

        // Bivariate Analysis Section
        const bivariateAnalysisHTML = `
            <div class="card">
                <h3>Bivariate Analysis</h3>
                <img src="data:image/png;base64,${edaData.bivariate_analysis.correlation_matrix}" alt="Correlation Matrix">
            </div>
        `;

        // Normalization Analysis Section
        const normalizationAnalysisHTML = `
            <div class="card">
                <h3>Normalization Analysis</h3>
                ${
                    edaData.normality_skewness_analysis
                        ? Object.keys(edaData.normality_skewness_analysis)
                              .map((col) => {
                                  const colData = edaData.normality_skewness_analysis[col];
                                  return `
                                      <h4>${col}</h4>
                                      <p><strong>Skewness:</strong> ${colData.skewness.toFixed(2)}</p>
                                      <p><strong>Normality Test p-value:</strong> ${colData.normality_test_pvalue.toFixed(4)}</p>
                                      <p><strong>Is Normal:</strong> ${colData.is_normal ? "Yes" : "No"}</p>
                                      <img src="data:image/png;base64,${colData.distribution_plot}" alt="${col} Distribution Plot">
                                  `;
                              })
                              .join("")
                        : "<p>No normalization data available.</p>"
                }
            </div>
        `;

        // Build the full HTML with sections
        reportContainer.innerHTML = `
            <div class="card">
                <h3>Dataset Overview</h3>
                ${datasetOverviewTable}
            </div>

            <div class="card">
                <h3>Descriptive Statistics</h3>
                ${descriptiveStatisticsTable}
            </div>

            ${dataQualityAnalysisHTML}
            ${univariateAnalysisHTML}
            ${bivariateAnalysisHTML}
            ${normalizationAnalysisHTML}
        `;
    } else {
        reportContainer.innerHTML = "<p>No EDA data found. Please upload a CSV file first.</p>";
    }
});
