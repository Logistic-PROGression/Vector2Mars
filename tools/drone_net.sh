#!/usr/bin/env bash
# Switch this laptop between the home network and the Pioneer Mini 2 Wi-Fi AP.
#
# The drone's AP runs on 5 GHz while the built-in Realtek RTL8821CE has a single
# radio, so both networks cannot be joined at once. This script makes the switch
# cheap and, more importantly, keeps the VPN from swallowing the drone route:
# happ-xray installs `default metric 1`, which wins over the Wi-Fi default route
# and also captures 10.42.0.0/24. A more specific /24 route pinned to the Wi-Fi
# interface fixes that without touching the VPN.
#
# Usage:
#   tools/drone_net.sh status      # where does drone traffic actually go?
#   tools/drone_net.sh connect     # join the drone AP and pin its route
#   tools/drone_net.sh disconnect  # drop the route, return to the home network
#
# `connect` and `disconnect` need sudo only for the route change.

set -uo pipefail

DRONE_SSID="PMINI2-CCUu9yqQWFqkzCzJFNgOKQ"
DRONE_PASS="geoscan123"
DRONE_IP="10.42.0.1"
DRONE_SUBNET="10.42.0.0/24"
HOME_CONN="HUAWEI 200"
PORTS=(9999 9090 22)

die() { printf 'Ошибка: %s\n' "$*" >&2; exit 1; }

wifi_iface() {
    local iface
    iface=$(nmcli -t -f DEVICE,TYPE,STATE device status \
        | awk -F: '$2=="wifi" && $3!="unmanaged" {print $1; exit}')
    [[ -n "$iface" ]] || die "не найден Wi-Fi интерфейс"
    printf '%s' "$iface"
}

# Return the interface the kernel would actually use for the drone address.
route_iface() {
    ip route get "$DRONE_IP" 2>/dev/null | awk '{for (i=1;i<=NF;i++) if ($i=="dev") {print $(i+1); exit}}'
}

route_is_pinned() {
    ip route show | grep -qE "^${DRONE_SUBNET//./\\.} dev "
}

port_open() {
    timeout 3 bash -c "exec 3<>/dev/tcp/${DRONE_IP}/$1" 2>/dev/null
}

print_ports() {
    local port
    for port in "${PORTS[@]}"; do
        if port_open "$port"; then
            printf '  порт %-5s доступен\n' "$port"
        else
            printf '  порт %-5s НЕ доступен\n' "$port"
        fi
    done
}

cmd_status() {
    local iface used
    iface=$(wifi_iface)
    used=$(route_iface)

    printf 'Wi-Fi интерфейс : %s\n' "$iface"
    printf 'Текущая сеть    : %s\n' "$(nmcli -t -f GENERAL.CONNECTION device show "$iface" | cut -d: -f2-)"
    printf 'Маршрут к %-8s: %s\n' "$DRONE_IP" "${used:-нет}"

    if [[ "$used" == "happ-xray" || "$used" == tun* || "$used" == wg* ]]; then
        printf '\nВНИМАНИЕ: трафик к дрону уходит в VPN (%s), а не в Wi-Fi.\n' "$used"
        printf 'Пинг и «открытые порты» в таком состоянии — ложное срабатывание VPN.\n'
        printf 'Исправляется командой: tools/drone_net.sh connect\n'
        return 1
    fi

    if [[ -z "$used" ]]; then
        printf '\nК дрону нет маршрута. Подключитесь: tools/drone_net.sh connect\n'
        return 1
    fi

    if ! route_is_pinned; then
        printf '\nМаршрут %s не закреплён за Wi-Fi — после переподключения его съест VPN.\n' "$DRONE_SUBNET"
    fi

    printf '\nПроверка портов:\n'
    print_ports
}

cmd_connect() {
    local iface
    iface=$(wifi_iface)

    printf 'Подключаюсь к дрону %s...\n' "$DRONE_SSID"
    printf 'Интернет на время работы с дроном пропадёт (один радиоблок).\n\n'

    nmcli device wifi connect "$DRONE_SSID" password "$DRONE_PASS" ifname "$iface" \
        || die "не удалось подключиться к $DRONE_SSID"

    # DHCP on the drone AP can take a moment; wait for a 10.42.0.x lease.
    local waited=0
    while (( waited < 15 )); do
        if ip -brief addr show "$iface" | grep -q '10\.42\.0\.'; then
            break
        fi
        sleep 1
        (( waited++ ))
    done

    printf '\nЗакрепляю маршрут %s за %s (обходит VPN)...\n' "$DRONE_SUBNET" "$iface"
    sudo ip route replace "$DRONE_SUBNET" dev "$iface" \
        || die "не удалось добавить маршрут (нужен sudo)"

    local used
    used=$(route_iface)
    if [[ "$used" != "$iface" ]]; then
        die "маршрут всё ещё идёт через $used — VPN перехватывает трафик"
    fi

    printf 'Готово: %s -> %s\n\n' "$DRONE_IP" "$used"
    print_ports
    printf '\nIDE:  http://%s:9999/?folder=/home/pioneermini/workspace\n' "$DRONE_IP"
    printf 'Jupyter: http://%s:9090/\n' "$DRONE_IP"
}

cmd_disconnect() {
    local iface
    iface=$(wifi_iface)

    if route_is_pinned; then
        printf 'Убираю маршрут %s...\n' "$DRONE_SUBNET"
        sudo ip route del "$DRONE_SUBNET" dev "$iface" 2>/dev/null \
            || sudo ip route del "$DRONE_SUBNET" 2>/dev/null \
            || printf 'Маршрут уже отсутствует.\n'
    fi

    printf 'Возвращаюсь в сеть «%s»...\n' "$HOME_CONN"
    nmcli connection up "$HOME_CONN" || die "не удалось вернуться в $HOME_CONN"

    printf 'Готово.\n'
}

case "${1:-status}" in
    status|check) cmd_status ;;
    connect|up)   cmd_connect ;;
    disconnect|down) cmd_disconnect ;;
    -h|--help|help)
        awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"
        ;;
    *) die "неизвестная команда: $1 (status | connect | disconnect)" ;;
esac
