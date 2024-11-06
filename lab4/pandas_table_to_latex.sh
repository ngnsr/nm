#!/bin/bash

input_file="program2.out"

output_file="output2.tex"

cat << EOF > "$output_file"
\documentclass{article}
\usepackage{amsmath}
\usepackage{array}
\begin{document}
\begin{table}[h!]
\hspace*{-2cm}
\begin{tabular}{|c|c|c|c|c|c|c|c|}
\hline
EOF

closeTableStartTable() {
    cat << EOF  >> "$output_file"
\end{tabular}
\end{table}

\begin{table}[h!]
\hspace*{-2cm}
\begin{tabular}{|c|c|c|c|c|c|c|c|}
\hline
EOF
}

closeTable() {
    cat << EOF >> "$output_file"
\end{tabular}
\end{table}
\end{document}
EOF
}

while IFS= read -r line; do
    if [[ -n "$line" ]]; then
        formatted_line=$(echo "$line" | sed -e 's/\\//g' -e 's/, \.\.\., /,\\cdots,/g' | tr -s ' ' '&' )
        printf '%s \\\\ \n' "${formatted_line::-1}" >> "$output_file"
        echo "\hline" >> "$output_file"
    else
        closeTableStartTable
    fi
done < "$input_file"

closeTable

echo "LaTeX таблиця збережена у файлі $output_file"

