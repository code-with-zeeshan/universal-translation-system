{{- define "uts.name" -}}
uts
{{- end -}}

{{- define "uts.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "uts.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "uts.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" -}}
{{- end -}}

{{- define "uts.image" -}}
{{- $img := . -}}
{{- printf "%s:%s" $img.repository $img.tag -}}
{{- end -}}