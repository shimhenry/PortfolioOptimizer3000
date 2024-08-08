from flask import Flask, render_template, request, send_file
import pandas as pd
from portfolio_optimizer import optimize_portfolio

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/optimize/<gamma>', methods=['POST'])
def optimize(gamma):
    print(gamma)
    if 'excel_file' not in request.files:
        return "No file part", 400
    file = request.files['excel_file']
    if file.filename == '':
        return "No selected file", 400
        return "No selected file", 400
    if file and file.filename.endswith('.xlsx'):
        input_df = pd.read_excel(file)

        # Print the columns of the DataFrame for diagnostics
        print("Columns in uploaded Excel file:", input_df.columns.tolist())

        # Strip leading/trailing spaces from column names
        input_df.columns = input_df.columns.str.strip()

        tickers = input_df['Tickers'].dropna().tolist()
        risk_tolerance = input_df['Risk_Tolerance'].dropna().iloc[0]
        principal = input_df['Principal(USD)'].dropna().iloc[0]
        risk_free_rate = input_df['RF Rate'].dropna().iloc[0]
        gamma = float(gamma)
        print(gamma)

        output_risk_aware, output_max_sharpe, performance_metrics_risk_aware, performance_metrics_max_sharpe = optimize_portfolio(
            tickers, risk_tolerance, principal, risk_free_rate, gamma)

        output_file_path = 'optimized_portfolio.xlsx'
        with pd.ExcelWriter(output_file_path) as writer:
            output_risk_aware.to_excel(writer, sheet_name='Risk Aware Portfolio', index=False)
            performance_metrics_risk_aware.to_excel(writer, sheet_name='Risk Aware Performance', index=False)
            output_max_sharpe.to_excel(writer, sheet_name='Max Sharpe Portfolio', index=False)
            performance_metrics_max_sharpe.to_excel(writer, sheet_name='Max Sharpe Performance', index=False)

        return send_file(output_file_path, as_attachment=True, download_name='optimized_portfolio.xlsx')
    else:
        return "Unsupported file type", 400

if __name__ == '__main__':
    app.run(debug=True)